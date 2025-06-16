import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.ssm.ssm_block import SSM_Block


class BBoxDetector(nn.Module):
    def __init__(self, num_bboxes: int, l_max: int):
        super().__init__()

        self.num_bboxes = num_bboxes
        self.l_max = l_max

        # Улучшенная обработка P3 (28x28)
        self.p3_conv1 = nn.Conv2d(192, 256, 3, padding=1)
        self.p3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Улучшенная обработка P4 (14x14)
        self.p4_conv1 = nn.Conv2d(384, 256, 3, padding=1)
        self.p4_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        # Обработка P5 (7x7) с последующим апсемплингом
        self.p5_conv1 = nn.Conv2d(576, 256, 3, padding=1)
        self.p5_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Объединение фичей + финальные слои
        self.fusion_conv = nn.Conv2d(768, 512, 3, padding=1)
        self.final_conv = nn.Conv2d(512, 256, 3, padding=1)

        # Head для предсказания bbox
        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_bboxes * 4, 1)  # 10 bbox * 4 координаты (cxcywh)
        )

        self.ssm_block_1 = SSM_Block(seq_len=l_max, layer_h=256)
        self.ssm_block_2 = SSM_Block(seq_len=l_max, layer_h=256)

        # Инициализация весов
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
        # Обработка P5 (7x7) -> 14x14
        p5 = F.relu(self.p5_conv1(p5))
        p5 = F.relu(self.p5_conv2(p5))
        p5_up = self.p5_upsample(p5)  # [B, 256, 14, 14]

        # Обработка P4 (14x14) и объединение с P5
        p4 = F.relu(self.p4_conv1(p4))
        p4 = F.relu(self.p4_conv2(p4))  # [B, 256, 14, 14]
        p4 = torch.cat([p4, p5_up], dim=1)  # [B, 512, 14, 14]

        # Обработка объединенного P4 и апсемплинг до 28x28
        p4_up = self.p3_upsample(p4)  # [B, 512, 28, 28]

        # Обработка P3 (28x28) и объединение с P4
        p3 = F.relu(self.p3_conv1(p3))
        p3 = F.relu(self.p3_conv2(p3))  # [B, 256, 28, 28]
        p3 = torch.cat([p3, p4_up], dim=1)  # [B, 768, 28, 28]

        # Финальное объединение и обработка
        x = F.relu(self.fusion_conv(p3))
        x = F.relu(self.final_conv(x))  # [B, 256, 28, 28]

        # попробуем вставить тут:
        x = self.ssm_block_1(x)
        x = self.ssm_block_2(x)

        # Предсказание bbox
        bbox = self.bbox_head(x)  # [B, 40, 28, 28]

        # Глобальный average pooling для получения [B, 40]
        bbox = F.adaptive_avg_pool2d(bbox, (1, 1)).flatten(1)

        # Reshape в [B, 10, 4]
        bbox = bbox.view(-1, self.num_bboxes, 4)

        # Активация для координат (sigmoid для cxcy, exp для wh)
        bbox[..., :2] = torch.sigmoid(bbox[..., :2])  # cx, cy в [0, 1]

        bbox[..., 2:] = torch.sigmoid(bbox[..., 2:])
        # too unstable
        # bbox[..., 2:] = torch.exp(bbox[..., 2:])  # width, height > 0

        return bbox
