import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import rfft, irfft
from scipy.linalg import eigh


def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum(dim=-1)


def K_gen_DPLR(Lambda, P, Q, B, C, step, L):
    aterm = (C.conj(), Q.conj())  # (C*, Q*)
    bterm = (B, P)  # (B, P)

    @torch.compiler.disable(recursive=False)
    def gen(z):
        g = (2.0 / step) * ((1.0 - z) / (1.0 + z))
        c = 2.0 / (1.0 + z)

        k00 = cauchy_dot(aterm[0] * bterm[0], g, Lambda)
        k01 = cauchy_dot(aterm[0] * bterm[1], g, Lambda)
        k10 = cauchy_dot(aterm[1] * bterm[0], g, Lambda)
        k11 = cauchy_dot(aterm[1] * bterm[1], g, Lambda)

        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def conv_from_gen(gen, L, device):
    Omega_L = torch.exp(-2j * torch.pi * torch.arange(L) / L).to(device)  # .to(torch.complex32)
    atRoots = torch.vmap(gen)(Omega_L)  # [L, N]
    out = torch.fft.ifft(atRoots, n=L, dim=0).real  # [L, N]
    return out


# ------------------------------------------------------------------------------
# подсчёт статичных значений

def make_HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float64))
    A = P.unsqueeze(1) * P.unsqueeze(0)
    A = torch.tril(A) - torch.diag(torch.arange(N, dtype=torch.float64))
    return -A


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = torch.sqrt(torch.arange(N, dtype=torch.float64) + 0.5)

    # HiPPO also specifies the B matrix
    B = torch.sqrt(2 * torch.arange(N, dtype=torch.float64) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P.unsqueeze(1) * P.unsqueeze(0)

    # Check skew symmetry
    S_diag = torch.diagonal(S)
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)
    # assert torch.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    # Note: eigh doesn't have a direct PyTorch equivalent, so we use scipy and convert back
    Lambda_imag, V = eigh(S.numpy() * -1j)
    Lambda_imag = torch.from_numpy(Lambda_imag)
    V = torch.from_numpy(V).type(torch.complex64)
    P = V.conj().T @ P.type(torch.complex64)
    B = V.conj().T @ B.type(torch.complex64)
    return (Lambda_real + 1j * Lambda_imag).type(torch.complex64), P, B, V


# ------------------------------------------------------------------------------

# @torch.compile
class SSM_Layer(nn.Module):
    x_state: torch.Tensor

    def __init__(self, layer_h: int, hidden_states: int, seq_len: int, learn_A=False) -> None:
        super().__init__()
        self.H = layer_h  # размерность входых значений
        self.N = hidden_states  # количество скрытых параметров на 1 входное значение
        self.L = seq_len  # длина последовательности

        predef_Lambda, predef_P, predef_B, _ = make_DPLR_HiPPO(self.N)
        predef_Lambda, predef_P, predef_B = \
            predef_Lambda.repeat(self.H, 1), predef_P.repeat(self.H, 1), predef_B.repeat(self.H, 1)

        # todo: попробовать задать малый шаг обучения. также если оставлять, то убрать дубликацию
        self._P = nn.Parameter(torch.view_as_real(predef_P))
        self._Lambda = nn.Parameter(torch.view_as_real(predef_Lambda))
        self.step = nn.Parameter(torch.Tensor([1 / self.L]))

        self._P.requires_grad_(learn_A)
        self._Lambda.requires_grad_(learn_A)
        self.step.requires_grad_(learn_A)

        self._B = nn.Parameter(torch.view_as_real(predef_B))
        self._C = nn.Parameter(torch.view_as_real(torch.randn(self.H, self.N, dtype=torch.complex64)))
        self._D = nn.Parameter(torch.randn(self.H, 1))

        self.cnn_mode = True
        self.naive_repr = None

        # self.register_buffer("x_state", tensor=torch.zeros(self.H, 3, self.N, dtype=torch.complex64))

    # -----------------------------------------------------------------------------

    @property
    def p(self) -> torch.Tensor:

        # assert torch.complex(self._P[:, :, 0], self._P[:, :, 0]) == torch.view_as_complex(self._P)
        return torch.view_as_complex(self._P)

    @property
    def lambda_(self) -> torch.Tensor:
        return torch.view_as_complex(self._Lambda)

    @property
    def B(self) -> torch.Tensor:
        return torch.view_as_complex(self._B)

    @property
    def C(self) -> torch.Tensor:
        return torch.view_as_complex(self._C)

    @property
    def D(self) -> torch.Tensor:
        return self._D

    # ---------------------------------------------------------------------------

    def set_CNN_mode(self):
        self.cnn_mode = True

    def set_RNN_mode(self, parallel_width: int, device="cuda"):
        self.init_naive(device=device)
        self.register_buffer("x_state",
                             tensor=torch.zeros(self.H, parallel_width, self.N, dtype=torch.complex64, device=device))
        self.cnn_mode = False

    def clear_x_state(self):
        self.x_state = torch.zeros(*self.x_state.shape, dtype=torch.complex64)

    # ---------------------------------------------------------------------------

    def init_naive_scalar(self, Lambda, P, Q, B, C):
        B = B.unsqueeze(-1)  # [N, 1]
        Ct = C.unsqueeze(0)  # [1, N]

        A = torch.diag(Lambda) - P.unsqueeze(-1) @ Q.unsqueeze(-1).conj().transpose(-2, -1)
        I = torch.eye(self.N, dtype=Lambda.dtype, device=Lambda.device)

        A0 = (2.0 / self.step) * I + A

        D = torch.diag(1.0 / ((2.0 / self.step) - Lambda))
        Qc = Q.conj().reshape(1, -1)
        P2 = P.reshape(-1, 1)
        intermediate = Qc @ (D @ P2)
        A1 = D - (D @ P2) * (1.0 / (1 + intermediate)) @ (Qc @ D)

        Ab = A1 @ A0
        Bb = (A1 @ B) * 2

        Ab_power = torch.matrix_power(Ab, self.L)
        Cb = (Ct @ torch.inverse(I - Ab_power).conj()).conj()

        Bb, Cb = Bb.squeeze(1), Cb.squeeze(0)
        return [Ab, Bb, Cb]

    # расчёт матриц A, B, C для всех H
    def init_naive(self, device="cuda"):
        n_AbBbCb = torch.vmap(self.init_naive_scalar, in_dims=(0, 0, 0, 0, 0))(self.lambda_, self.p, self.p, self.B,
                                                                               self.C)
        n_AbBbCb = list(el.to(device) for el in n_AbBbCb)

        max_eig_across_As = max(torch.linalg.eigvals(A).real.max().item() for A in n_AbBbCb[0])

        if max_eig_across_As >= 1:  # heuristic for discrete form
            raise RuntimeError(f"one of As is unstable! (max eig val across all As is {max_eig_across_As})")
        if max_eig_across_As >= 1 - 1e-4:
            import warnings
            warnings.warn(f"one of As is near stability boundary! (max eig val across all As is {max_eig_across_As})")

        self.naive_repr = n_AbBbCb  # .detach()

    def check_stability(self):
        temp_AbBbCb = torch.vmap(self.init_naive_scalar, in_dims=(0, 0, 0, 0, 0))(self.lambda_, self.p, self.p, self.B,
                                                                                  self.C)

        max_eig_across_As = max(torch.linalg.eigvals(A).real.max().item() for A in temp_AbBbCb[0])

        if max_eig_across_As >= 1:
            raise RuntimeError(f"one of As is unstable! (max eig val across all As is {max_eig_across_As})")
        if max_eig_across_As >= 1 - 1e-4:
            import warnings
            warnings.warn(f"one of As is near stability boundary! (max eig val across all As is {max_eig_across_As})")
        else:
            print(f"As is stable (max eig val across all As is {max_eig_across_As})")

    # ---------------------------------------------------------------------------

    # CNN forward для 1 скалярной величины с поддержкой батчей
    def forward_batched_scalar(self, lambda_, p, q, B, C, u):
        gen_func = K_gen_DPLR(lambda_, p, q, B, C, step=self.step, L=self.L)
        kernel = conv_from_gen(gen_func, self.L, lambda_.device)

        u_pad = F.pad(u, (0, kernel.shape[-1] - 1))
        k_pad = F.pad(kernel, (0, u.shape[-1] - 1))

        u_fft, k_fft = rfft(u_pad), rfft(k_pad)

        y = u_fft * k_fft.unsqueeze(0)
        y = irfft(y, n=kernel.shape[-1] + u.shape[-1] - 1)
        return y[:, :self.L]

    # RNN forward для 1 скалярной величины
    # с поддержкой одинаковой обработки параллельных потоков данных
    def forward_naive_scalar(self, A, B, C, x, u):

        def _scalar(_x, _u):
            _x = (A @ _x).contiguous() + (B * _u).contiguous()
            _y = C @ _x
            return _y, _x

        y, x = torch.vmap(_scalar, in_dims=(0, 0))(x, u)

        return y, x

    # ---------------------------------------------------------------------------

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u [BATCH, SEQ_LEN, D_INPUT]

        if self.cnn_mode:
            # обработка D_INPUT сиквенсов [BATCH, SEQ_LEN]
            _y_no_D = torch.vmap(self.forward_batched_scalar, in_dims=(0, 0, 0, 0, 0, 2))(self.lambda_, self.p, self.p,
                                                                                          self.B, self.C, u)
            _y_no_D = _y_no_D.permute(1, 2, 0)  # результат к виду [BATCH, SEQ_LEN, D_INPUT]
            # _y_no_D = torch.stack([
            #     self.forward_batched_scalar(self.lambda_[i], self.p[i], self.p[i], self.B[i], self.C[i], u[..., i])
            #     for i in range(u.shape[2])
            # ], dim=-1)

            return _y_no_D + u @ self.D  # + прямая связь
        else:
            if u.shape[1] != 1:  # todo: добавить проверку на размеры батча u.shape[0] != 1
                raise ValueError('RNN support only seq_len=1. [b, 1, H]')
            if self.naive_repr is None:
                raise RuntimeError('Ab, Bb, Cb is not defined!')

            Ab_h, Bb_h, Cb_h = self.naive_repr

            u = u.squeeze(1)  # remove seq dims

            # todo: перефразировать нормально
            # vmap по H скалярным величинам, одинаковый для каждого батча (у каждого батча своя память)
            # т.е. это для оптимизации и ухода от bs одинаковых SSM_Layer для процессинга bs параллельных потоков
            _y_no_D, self.x_state = torch.vmap(self.forward_naive_scalar, in_dims=(0, 0, 0, 0, 1))(Ab_h, Bb_h, Cb_h,
                                                                                                   self.x_state, u)
            _y_no_D = _y_no_D.permute(1, 0).reshape(u.shape)

            return (_y_no_D.real + u @ self.D).unsqueeze(1)  # return seq dims
