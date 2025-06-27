import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.ssm.ssm_block import SSM_Block


class Neck(nn.Module):
    def __init__(self, num_bboxes: int, num_classes: int, l_max: int):
        super().__init__()
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.l_max = l_max

        # Обработка P3 (28x28)
        self.p3_conv1 = nn.Conv2d(192, 256, 3, padding=1)
        #self.p3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Обработка P4 (14x14)
        self.p4_conv1 = nn.Conv2d(384, 256, 3, padding=1)
        #self.p4_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        # Обработка P5 (7x7)
        self.p5_conv1 = nn.Conv2d(576, 256, 3, padding=1)
        #self.p5_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Объединение фичей
        self.fusion_conv = nn.Conv2d(768, 512, 3, padding=1)
        self.final_conv = nn.Conv2d(512, 256, 3, padding=1)

        self.intermediate_conv = nn.Conv2d(256, 256, 3, padding=1)


        self.ssm_block_1 = SSM_Block(seq_len=l_max, in_shape=(256, 28, 28))
        self.ssm_block_2 = SSM_Block(seq_len=l_max, in_shape=(256, 28, 28))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_stability(self):
        self.ssm_block_1.check_stability()
        self.ssm_block_2.check_stability()

    def forward(self, p3, p4, p5):
        # Обработка P5 -> 14x14
        p5 = F.relu(self.p5_conv1(p5))
        #p5 = F.relu(self.p5_conv2(p5))
        p5_up = self.p5_upsample(p5)  # [B, 256, 14, 14]

        # Обработка P4 + объединение с P5
        p4 = F.relu(self.p4_conv1(p4))
        #p4 = F.relu(self.p4_conv2(p4))  # [B, 256, 14, 14]
        p4 = torch.cat([p4, p5_up], dim=1)  # [B, 512, 14, 14]
        p4_up = self.p3_upsample(p4)  # [B, 512, 28, 28]

        # Обработка P3 + объединение с P4
        p3 = F.relu(self.p3_conv1(p3))
        #p3 = F.relu(self.p3_conv2(p3))  # [B, 256, 28, 28]
        p3 = torch.cat([p3, p4_up], dim=1)  # [B, 768, 28, 28]

        # Финальные слои
        x = F.relu(self.fusion_conv(p3))
        x = F.relu(self.final_conv(x))  # [B, 256, 28, 28]

        # попробуем вставить тут:
        x = self.ssm_block_1(x)

        x = F.relu(self.intermediate_conv(x))

        x = self.ssm_block_2(x)

        return x
