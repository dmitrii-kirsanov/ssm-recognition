import torch

from core.model.ssm.ssm_layer import SSM_Layer

#todo: проверить, что будет если обучать (на нормированных и не нормированных данных
if __name__ == "__main__":
    import time

    seq_len = 2000
    h, n = 3, 8


    u = torch.randn(1, seq_len, h).cuda() * 2
    layer = SSM_Layer(layer_h=h, hidden_states=n, seq_len=seq_len).cuda()

    #print(layer._Lambda)


    # wh = 2
    # ibs = 1
    # with torch.no_grad():
    #     layer = SSM_Layer(layer_h=h, hidden_states=n, l_max=seq_len).cuda()
    #     layer.compile()
    #
    #     _p3 = torch.randn(ibs * seq_len, h, wh, wh).cuda()
    #     print(_p3)
    #     print("----------------------")
    #     l_max = seq_len
    #
    #     p3 = _p3.clone()
    #     bs, ch, h, w = p3.shape
    #     p3 = p3.reshape(bs // l_max, l_max, ch, h, w)
    #     p3 = p3.permute(0, 3, 4, 1, 2)  # b, h, w, seq, ch
    #     p3 = p3.reshape(-1, l_max, ch)
    #     with torch.amp.autocast('cuda', enabled=False):
    #         p3 = p3.to(torch.float32)
    #     #    print(p3)
    #         p3 = layer(p3)
    #     p3 = p3.reshape(bs // l_max, h, w, ch, l_max)
    #     p3 = p3.permute(0, 3, 4, 1, 2)
    #     p3 = p3.reshape(-1, ch, h, w)
    #     print(p3)
    #     print("-------------------------------------------------")
    #
    #     layer.set_RNN_mode(wh * wh)
    #     _p3 = _p3.reshape(ibs, seq_len, h, wh, wh)[0]
    #     bs, l_max = 1, 1
    #     for i in range(_p3.shape[0]):
    #         p3 = _p3[i].clone()
    #
    #         p3 = p3.reshape(bs // l_max, l_max, ch, h, w)
    #         p3 = p3.permute(0, 3, 4, 1, 2)  # b, h, w, seq, ch
    #         p3 = p3.reshape(-1, l_max, ch)
    #         with torch.amp.autocast('cuda', enabled=False):
    #             p3 = p3.to(torch.float32)
    #         #    print(p3)
    #             p3 = layer(p3)
    #         p3 = p3.reshape(bs // l_max, h, w, ch, l_max)
    #         p3 = p3.permute(0, 3, 4, 1, 2)
    #         p3 = p3.reshape(-1, ch, h, w)
    #         print(p3)


    print(f"{seq_len=}, {h=}, {n=}")

    print("-------------------------------------------------")
    _st = time.time()
    #with torch.no_grad():
    with torch.amp.autocast('cuda'):
        print(u.dtype)
        with torch.amp.autocast('cuda', enabled=False):
            res = layer(u)
        #print(res.dtype)

    print(res.detach().cpu().numpy()[-1, -1, :])
    print(f"est CNN time: {time.time() - _st}")

    print("-------------------------------------------------")

    layer.set_RNN_mode(1)
    _st = time.time()
    with torch.no_grad():
        for i in range(seq_len):
            _u = u[:, i, :].unsqueeze(0)  # [1, 1, H]
            res = layer(_u.repeat(1, 1, 1)) # [2, 1, H]
        print(res.cpu().numpy())
    print(f"est RNN time: {time.time() - _st}")
    print("-------------------------------------------------")