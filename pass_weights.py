import torch
from core.model.model import Model

OLD_WEIGHTS_PATH = "./data/weights/v_1.02_e_001_e_0.7889.pth"
NEW_WEIGHTS_PATH = "./data/weights/moved_v_1.02_e_001_e_0.7889.pth"

if __name__ == "__main__":
    model = Model()

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