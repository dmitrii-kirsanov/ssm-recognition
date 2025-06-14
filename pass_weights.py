import torch
from core.model.model import Model

OLD_WEIGHTS_PATH = "./data/weights/experimental_ssm_v2.20_no_blocks_iou_0.8276_cl_6.3292.pth"
NEW_WEIGHTS_PATH = "./data/weights/experimental_ssm_v2.20_clear_ssm_blocks.pth"

if __name__ == "__main__":
    model = Model(seq_len=240*1, num_pred=5, num_classes=10)

    state_dict = model.state_dict()
    pretrain_state_dict = torch.load(OLD_WEIGHTS_PATH, weights_only=True, map_location=torch.device('cpu'))

    for param_name, value in pretrain_state_dict.items():
        if param_name not in state_dict:
            continue
        state_dict[param_name] = value

    model.load_state_dict(state_dict)

    torch.save(
        model.state_dict(),
        NEW_WEIGHTS_PATH
    )