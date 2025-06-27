import torch
from absl.testing.parameterized import parameters
from torch import nn

from core.model.ssm.ssm_layer import SSM_Layer


class SSM_Block(nn.Module):

    def __init__(self, seq_len: int, in_shape: tuple[int, int, int], hidden_states=64, dropout=0.025):
        super().__init__()

        self.channels, self.w, self.h = in_shape

        self._seq_len = seq_len
        self._is_mode_cnn = True

        self.ssm_layer = SSM_Layer(layer_h=self.channels, hidden_states=hidden_states, seq_len=seq_len)
        self.drop = nn.Dropout(p=dropout)

        #self.linear = nn.Linear(self.channels, self.channels)

        self.in_norm = nn.LayerNorm(self.channels)
        self.out_norm1 = nn.LayerNorm(self.w * self.h)
        self.out_norm2 = nn.LayerNorm(self.w * self.h)

    @property
    def seq_len(self):
        return self._seq_len if self._is_mode_cnn else 1

    # todo: handle "to CNN" mode
    def set_mode(self, to_rnn: bool, device):
        self._is_mode_cnn = not to_rnn
        self.ssm_layer.set_RNN_mode(parallel_width=self.w * self.h, device=device)

    def check_stability(self):
        print(f"stability of {self.__hash__()}:")
        self.ssm_layer.check_stability()

    def pre_transform(self, p, shape):  # [bs * seq, c, w, h] -> [bs * h * w, seq, c]
        bs, ch, h, w = shape
        p = p.reshape(bs // self.seq_len, self.seq_len, ch, h, w)  # bs, seq, c, h, w
        p = p.permute(0, 3, 4, 1, 2)  # bs, h, w, seq, c
        p = p.reshape(-1, self.seq_len, ch)  # [bs * h * w, seq, c]
        return p

    def post_transform(self, p, shape):  # [bs * h * w, seq, c] -> [bs * seq, c, w, h]
        bs, ch, h, w = shape
        p = p.reshape(bs // self.seq_len, h, w, ch, self.seq_len)  # [bs, h, w, seq, c]
        p = p.permute(0, 3, 4, 1, 2)  # [bs, seq, c, h, w,]
        p = p.reshape(-1, ch, h, w)  # [bs * seq, c, h, w,]
        return p

    def forward(self, ix):  # x: [bs, c, w, h]
        input_shape = ix.shape
        x = self.pre_transform(ix, input_shape)  # [bs * h * w, seq, c]
        x = self.in_norm(x)

        with torch.amp.autocast(str(x.device), enabled=False):
            x = self.ssm_layer(x.float())

        x = self.drop(nn.functional.gelu(x))
        #x = self.linear(x)
        x = x * nn.functional.sigmoid(x)

        #x = self.drop(x)
        x = self.post_transform(x, input_shape)  # x: [bs, c, w, h]

        # normalize across w*h
        bs, c, w, h = input_shape
        x, ix = x.reshape(bs, c, w * h), ix.reshape(bs, c, w * h)
        x, ix = self.out_norm1(x), self.out_norm2(ix)
        x, ix = x.reshape(bs, c, w, h), ix.reshape(bs, c, w, h)

        ix = self.drop(ix)
        x = x + ix

        return x
