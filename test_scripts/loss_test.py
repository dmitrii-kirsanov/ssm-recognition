import torch

from core.loss.loss import loss_v2
from core.util.draw_output import show_new_img_bbox


def simple_search(num_iter=300):
    example_scale = 1
    image = torch.ones(3, 224 * example_scale, 224 * example_scale) * 255

    outputb = torch.nn.Parameter(torch.rand(2, 10, 10 + 4))
    targetb = torch.cat((torch.rand(2, 10, 4),
                         torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(0).repeat(2, 10, 1)),
                        dim=-1)
    targetb[:, :, 0:2] = torch.rand(2, 10, 2) * 2 - 1
    targetb[0, -2:, 4:] = 0

    optimizer = torch.optim.Adam([outputb], lr=0.006)

    def draw(scale=2, debug_negative=True, save_path=None):
        with torch.no_grad():
            showed_targetb = targetb.clone()
            showed_targetb[:, :, 0:2] = (showed_targetb[:, :, 0:2] + scale / (1 if debug_negative else 2)) / (scale * 2)
            showed_targetb[:, :, 2:4] = showed_targetb[:, :, 2:4] / (scale * 2)
            showed_outputb = outputb.clone()
            showed_outputb[:, :, 0:2] = (showed_outputb[:, :, 0:2] + scale / (1 if debug_negative else 2)) / (scale * 2)
            showed_outputb[:, :, 2:4] = showed_outputb[:, :, 2:4] / (scale * 2)

            for i in range(showed_targetb.shape[0]):
                show_new_img_bbox(image, showed_targetb[i], showed_outputb[i], save_path=save_path)
                if save_path:  # сохраняем ток из первого батча
                    break

    draw()

    for i in range(num_iter):
        loss, avg_iou_err, _, _, _ = loss_v2(outputb, targetb,  num_classes=10,
                                                         k_bbox_loss=0.8, k_class_loss=0.2,
                                                         threshold_iou_for_class_loss=0.75)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(avg_iou_err.item())
        #draw(save_path=f"/home/dima/Projects/ssm-recognition/data/experiments/loss/img_{i:06d}.png")

    draw()


def simple_example():
    example_scale = 1
    image = torch.ones(3, 224 * example_scale, 224 * example_scale) * 255

    outputb = torch.tensor([[
        [0.1, 0.1, 0.1, 0.1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0.8, 0.2, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0.3, 0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]])
    targetb = torch.tensor([[
        [0.2, 0.2, 0.2, 0.2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0.4, 0.7, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]])

    outputb.requires_grad = True

    loss, iou_score, _, _, _ = loss_v2(outputb, targetb, num_classes=10)

    # print(iou_score)

    show_new_img_bbox(image, targetb.squeeze(0), outputb.squeeze(0))


if __name__ == "__main__":
    # simple_example()
    simple_search()
    # exit(0)