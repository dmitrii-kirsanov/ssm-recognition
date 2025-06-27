import os

import cv2
import torch

from core.model.backbone.backbone import Backbone2D
from core.model.experimental.classifier import Classifier
from core.model.experimental.localizator import Localizator
from core.model.experimental.neck import Neck


@torch.compile
class Model(torch.nn.Module):
    def __init__(self, seq_len: int, num_pred: int, num_classes: int):
        super().__init__()

        self.seq_len = seq_len
        self.num_pred = num_pred
        self.num_classes = num_classes

        self.backbone = Backbone2D()
        self.neck = Neck(num_bboxes=num_pred, num_classes=num_classes, l_max=seq_len)
        self.localization_head = Localizator(num_bboxes=num_pred)
        self.classification_head = Classifier(num_bboxes=num_pred, num_classes=num_classes)


    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        B, SL, C, H, W = x.shape
        x = x.view(B * SL, 3, H, W)
        p3, p4, p5 = self.backbone(x)

        x = self.neck(p3, p4, p5)

        x_bbox = self.localization_head(x)
        x_class = self.classification_head(x)

        x = torch.cat((x_bbox, x_class), dim=-1)
        x = x.view(B, SL, self.num_pred, 4 + self.num_classes)

        return x

    def check_stability(self):
        self.neck.check_stability()
        # ...

    def inference(self, cv2_input_image):
        with (torch.no_grad()):
            _img = cv2.resize(cv2_input_image, (224, 224))

            img_tensor = torch.Tensor(_img).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

            img_tensor[:, :, 0, :, :] = (img_tensor[:, :, 0, :, :] / 255.0 - 0.485) / 0.229  # R
            img_tensor[:, :, 1, :, :] = (img_tensor[:, :, 1, :, :] / 255.0 - 0.456) / 0.224  # G
            img_tensor[:, :, 2, :, :] = (img_tensor[:, :, 2, :, :] / 255.0 - 0.485) / 0.225  # B

            output = self(img_tensor)
            output = output.squeeze(0).squeeze(0)

            bboxes, classes = output.split((4, self.num_classes), -1)
            classes = torch.nn.functional.sigmoid(classes)
            output = torch.cat((bboxes, classes), dim=-1)

            return output

    def load_pretrain_backbone(self, pretrain_path):
        self.backbone.load_pretrain(pretrain_path)

    def freeze_backbone(self, state: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not state

    def freeze_localization_head(self, state: bool):
        for p in self.localization_head.parameters():
            p.requires_grad = not state

    def freeze_classification_head(self, state: bool):
        for p in self.classification_head.parameters():
            p.requires_grad = not state

    def load(self, weights_filename, strict=True, WEIGHTS_PATH="./data/weights"):
        self.load_state_dict(torch.load(
            os.path.join(WEIGHTS_PATH, weights_filename),
            weights_only=True), strict=strict
        )

    def save(self, weights_filename, WEIGHTS_PATH="./data/weights"):
        torch.save(
            self.state_dict(),
            os.path.join(WEIGHTS_PATH, weights_filename)
        )
