import torch
from core.model.backbone.backbone_legacy import DarkNet, DarkFPN

#@torch.compile
class Backbone2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #m
        depth = [2, 4, 4]
        width = [3, 48, 96, 192, 384, 576]

        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)


    def forward(self, x):
        x = self.net(x)
        return self.fpn(x)

    def load_pretrain(self, pretrain_path):
        state_dict = self.state_dict()

        pretrain_state_dict = torch.load(pretrain_path, weights_only=True, map_location=torch.device('cpu')) #todo: handle it

        i_good, i_bad = 0, 0
        for param_name, value in pretrain_state_dict.items():
            if param_name not in state_dict:
                i_bad += 1
                continue
            i_good += 1
            #print(f"{param_name}")
            state_dict[param_name] = value

        self.load_state_dict(state_dict)

        print(f"backbone2D : YOLOv8 pretrained loaded! (loaded={i_good}, skipped={i_bad})", flush=True)