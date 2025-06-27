import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, num_bboxes: int, num_classes: int):
        super().__init__()

        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        linear_factor = 1
        # Head для предсказания bbox
        self.sequential_main = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(256, 256, 3, padding=1),
            #nn.ReLU(),
            nn.Conv2d(256, num_bboxes * num_classes * linear_factor, 1)  # num_bboxes * 4 координаты (cxcywh)
        )

        #self.linear = nn.Linear(num_bboxes * num_classes * linear_factor, num_bboxes * num_classes)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        logits = self.sequential_main(x)  # x - [B, num_bboxes * num_classes, 28, 28]
        logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(1)  # [B, num_bboxes*num_classes]

        #logits = self.linear(logits)
        logits = logits.view(-1, self.num_bboxes, self.num_classes)  # [B, num_bboxes, num_classes]

        return logits
