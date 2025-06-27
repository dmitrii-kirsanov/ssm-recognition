import torch
import torch.nn as nn
import torch.nn.functional as F


class Localizator(nn.Module):
    def __init__(self, num_bboxes: int):
        super().__init__()

        self.num_bboxes = num_bboxes

        linear_factor = 1
        # Head для предсказания bbox
        self.sequential_main = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(256, 256, 3, padding=1),
            #nn.ReLU(),
            nn.Conv2d(256, num_bboxes * 4 * linear_factor, 1)  # num_bboxes * 4 координаты (cxcywh)
        )

        #self.linear = nn.Linear(num_bboxes * 4 * linear_factor, num_bboxes * 4)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        bbox = self.sequential_main(x)  # x - [B, num_bboxes * 4, 28, 28]
        bbox = F.adaptive_avg_pool2d(bbox, (1, 1)).flatten(1)  # [B, num_bboxes*4]

        #bbox = self.linear(bbox)
        bbox = bbox.view(-1, self.num_bboxes, 4)  # [B, num_bboxes, 4]

        # Активация для координат (sigmoid для cxcy, sigmoid для wh)
        bbox[..., :2] = torch.sigmoid(bbox[..., :2])  # cx, cy в [0, 1]
        bbox[..., 2:] = torch.sigmoid(bbox[..., 2:])

        return bbox
