import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import fft, ifft, rfft, irfft
from scipy.linalg import eigh
from torch.nn.functional import mse_loss


def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum(dim=-1)

def K_gen_DPLR(Lambda, P, Q, B, C, step, L):
    aterm = (C.conj(), Q.conj())  # (C*, Q*)
    bterm = (B, P)                # (B, P)

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
    Omega_L = torch.exp(-2j * torch.pi * torch.arange(L) / L).to(device)#.to(torch.complex32)
    atRoots = torch.vmap(gen)(Omega_L)  # [L, N]
    out = torch.fft.ifft(atRoots, n=L, dim=0).real  # [L, N]
    return out


#------------------------------------------------------------------------------
#подсчёт статичных значений

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

#------------------------------------------------------------------------------


class SSM_Layer(nn.Module):

    x_state: torch.Tensor

    def __init__(self, layer_h: int, hidden_states: int, l_max: int) -> None:
        super().__init__()
        self.H = layer_h #размерность входых значений
        self.N = hidden_states #количество скрытых параметров на 1 входное значение
        self.L = l_max #длина последовательности

        predef_Lambda, predef_P, predef_B, _ = make_DPLR_HiPPO(self.N)
        predef_Lambda, predef_P, predef_B = predef_Lambda.repeat(self.H, 1), predef_P.repeat(self.H, 1), predef_B.repeat(self.H, 1)

        self._P = nn.Parameter(torch.view_as_real(predef_P))
        self._Lambda = nn.Parameter(torch.view_as_real(predef_Lambda))

        self._B = nn.Parameter(torch.view_as_real(predef_B))
        self._C = nn.Parameter(torch.view_as_real(torch.randn(self.H, self.N, dtype=torch.complex64)))
        self._D = nn.Parameter(torch.randn(self.H, 1))
        self.step = nn.Parameter(torch.Tensor([1 / self.L]))

        self.cnn_mode = True
        self.naive_repr = None

        self.register_buffer("x_state", tensor=torch.zeros(self.H, self.N, dtype=torch.complex64))

    #-----------------------------------------------------------------------------

    @property
    def p(self) -> torch.Tensor:

        #assert torch.complex(self._P[:, :, 0], self._P[:, :, 0]) == torch.view_as_complex(self._P)
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

    #---------------------------------------------------------------------------

    def set_CNN_mode(self):
        self.cnn_mode = True

    def set_RNN_mode(self):
        self.init_naive()
        self.cnn_mode = False

    def clear_x_state(self):
        self.x_state = torch.zeros(self.H, self.N, dtype=torch.complex64)

    #---------------------------------------------------------------------------

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
        Cb =(Ct @ torch.inverse(I - Ab_power).conj()).conj()

        Bb, Cb = Bb.squeeze(1), Cb.squeeze(0)
        return [Ab, Bb, Cb]


    #расчёт матриц A, B, C для всех H
    def init_naive(self):
        n_AbBbCb = torch.vmap(self.init_naive_scalar, in_dims=(0, 0, 0, 0, 0))(self.lambda_, self.p, self.p, self.B, self.C)
        self.naive_repr = n_AbBbCb #.detach()

    #---------------------------------------------------------------------------

    #CNN forward для 1 скалярной величины с поддержкой батчей

    #@torch.compiler.disable(recursive=False)
    def forward_batched_scalar(self, lambda_, p, q, B, C, u):
        gen_func = K_gen_DPLR(lambda_, p, q, B, C, step=self.step, L=self.L)
        kernel = conv_from_gen(gen_func, self.L, lambda_.device)

        u_pad =  F.pad(u, (0, kernel.shape[-1] - 1))
        k_pad =  F.pad(kernel, (0, u.shape[-1] - 1))

        u_fft, k_fft = rfft(u_pad), rfft(k_pad)

        y = u_fft * k_fft.unsqueeze(0)
        y = irfft(y, n=kernel.shape[-1] + u.shape[-1] - 1)
        return y[:, :self.L]


    #RNN forward для 1 скалярной величины
    def forward_naive_scalar(self, A, B, C, x, u):
        x = A @ x + B * u
        y = C @ x
        return y, x

    #---------------------------------------------------------------------------

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        #u [BATCH, SEQ_LEN, D_INPUT]

        if self.cnn_mode:
            #обработка D_INPUT сиквенсов [BATCH, SEQ_LEN]
            _y_no_D = torch.vmap(self.forward_batched_scalar, in_dims=(0, 0, 0, 0, 0, 2))(self.lambda_, self.p, self.p, self.B, self.C, u)
            _y_no_D = _y_no_D.permute(1, 2, 0) #результат к виду [BATCH, SEQ_LEN, D_INPUT]
            return _y_no_D + u @ self.D # + прямая связь
        else:
            if u.shape[0] != 1 or u.shape[1] != 1:
                raise ValueError('RNN support only batch=1 и seq_len=1. [1, 1, H]')
            if self.naive_repr == None:
                raise RuntimeError('Ab, Bb, Cb is not defined!')

            Ab_h, Bb_h, Cb_h = self.naive_repr
            u = u.squeeze(0).squeeze(0) #remove batch и seq dims

            #vmap по H скалярным величинам
            _y_no_D, self.x_state = torch.vmap(self.forward_naive_scalar, in_dims=(0, 0, 0, 0, 0))(Ab_h, Bb_h, Cb_h, self.x_state, u)

            return (_y_no_D.real + u @ self.D).unsqueeze(0).unsqueeze(0) #return batch и seq dims


if __name__ == "__main__":
    import time

    seq_len = 32
    h, n = 2048, 8

    u = torch.randn(1, seq_len, h).cuda()
    # print(u) # [1, seq_len, h]

    layer = SSM_Layer(layer_h=h, hidden_states=n, l_max=seq_len).cuda()
    layer.compile()


    print(f"{seq_len=}, {h=}, {n=}")
    print("-------------------------------------------------")
    _st = time.time()
    #with torch.no_grad():
    res = layer(u)
    loss = mse_loss(res, torch.zeros(*res.shape).cuda())
    loss.backward()
        #print(res.cpu().numpy())
    print(f"est CNN time: {time.time() - _st}")



    # print("-------------------------------------------------")
    # layer.set_RNN_mode()
    # _st = time.time()
    # with torch.no_grad():
    #     for i in range(seq_len):
    #         _u = u[:, i, :].unsqueeze(0)  # [1, 1, H]
    #         res = layer(_u)
    #         #print(res.cpu().numpy())
    # print(f"est RNN time: {time.time() - _st}")
    # print("-------------------------------------------------")