import torch

from core.model.backbone.backbone import Backbone2D
from core.model.experimental.bbox_classifier import BBoxClassifier
from core.model.experimental.bbox_detector import BBoxDetector


# from LRU_pytorch import LRU

@torch.compile
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone2D()  # medium size # freeze ???
        self.bbox_detector = BBoxDetector()
        self.bbox_classifier = BBoxClassifier()
        # self.training = True

    def forward(self, x):
        B, SL, C, H, W = x.shape
        x = x.view(B * SL, 3, H, W)
        p3, p4, p5 = self.backbone(x)

        x_bbox = self.bbox_detector(p3, p4, p5)  # pass shape here?
        x_class = self.bbox_classifier(p3, p4, p5)

        # if not self.training:
        #    x_class = torch.nn.functional.sigmoid(x_class)

        x = torch.cat((x_bbox, x_class), dim=-1)
        x = x.view(B, SL, 5, 4 + 10)

        return x

    def inference(self, img):
        pass  # сделать всю пред/пост -обработку тут

    def load_pretrain_backbone(self, pretrain_path):
        self.backbone.load_pretrain(pretrain_path)

    def freeze_backbone(self, state: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not state
