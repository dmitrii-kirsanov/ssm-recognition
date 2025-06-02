import torch
import tqdm
import os

from core.loss.loss import loss_v2
from core.model.model import Model
from core.util.custom_dataset import CustomDataset
from core.util.draw_output import show_new_img_bbox

torch.manual_seed(0)
torch.set_float32_matmul_precision('high')  # warning recommend to do it

DATA_PATH = "./data/ava/preprocessed_videos/"
PRETRAINED_WEIGHTS_PATH = "./data/weights"

model = Model()

model.load_state_dict(torch.load(
    os.path.join(PRETRAINED_WEIGHTS_PATH, "on_299_v_1.11_e_049_iou_0.8529.pth"),
    weights_only=True)
)

#model.load_pretrain_backbone("./data/pretrain/v8_m.pth")
model.freeze_backbone(True)
model.to("cuda")

num_epochs = 100
criterion = loss_v2
batch_size = 60
seconds_seq_len = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)  # 100 эпох
scaler = torch.amp.GradScaler('cuda')

cd = CustomDataset(DATA_PATH, fps=1, seconds_seq_len=seconds_seq_len, max_videos=299, classes_num=10, pred_max=5)
dataloader = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True,
                                         num_workers=0)  # todo: check collate_fn #num_workers was 8

for epoch_id in range(num_epochs):
    running_iou, running_class_loss = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    pbar = tqdm.tqdm(dataloader, desc="epoch training")
    for batch in pbar:
        images, targets = batch
        images, targets = images.to("cuda"), targets.to("cuda")
        # images [B, SL, 3, H, W], targets [B, SL, num_pred, 4(bb) + 10(classes)]

        # normalize: #todo: move to other location. check that's its actually R,G,B
        images[:, :, 0, :, :] = (images[:, :, 0, :, :] / 255.0 - 0.485) / 0.229  # R
        images[:, :, 1, :, :] = (images[:, :, 1, :, :] / 255.0 - 0.456) / 0.224  # G
        images[:, :, 2, :, :] = (images[:, :, 2, :, :] / 255.0 - 0.485) / 0.225  # B

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            #print(torch.max(outputs))
            b, sl, n, po = outputs.shape
            _, _, m, pt = targets.shape
            loss, avg_target_iou, class_loss = criterion(outputs.reshape(b * sl, n, po),
                                             targets.reshape(b * sl, m, pt))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_iou += avg_target_iou
        running_class_loss += class_loss

    avg_iou = running_iou.item() / len(dataloader)
    avg_class_loss = running_class_loss.item() / len(dataloader)
    print(f"epoch: {epoch_id:03d}, avg_target_iou: {avg_iou}, avg_class_loss: {avg_class_loss}")

    # visual validation
    with torch.no_grad():
        _images, _targets = next(iter(dataloader))

        images, targets = _images.clone().to("cuda"), _targets.clone().to("cuda")
        images[:, :, 0, :, :] = (images[:, :, 0, :, :] / 255.0 - 0.485) / 0.229  # R
        images[:, :, 1, :, :] = (images[:, :, 1, :, :] / 255.0 - 0.456) / 0.224  # G
        images[:, :, 2, :, :] = (images[:, :, 2, :, :] / 255.0 - 0.485) / 0.225  # B

        outputs = model(images)
        show_new_img_bbox(_images[0, 0], _targets[0, 0], outputs[0, 0], # если тут падает ошибка, то 99% выходные значения типа nan/inf
                          save_path=f"/home/dima/Projects/ssm-recognition/data/experiments/train/img_e_{epoch_id}.png")

    torch.save(
        model.state_dict(),
        os.path.join(PRETRAINED_WEIGHTS_PATH, f"v_1.11_e_{epoch_id:03d}_iou_{avg_iou:4.4f}.pth")
    )
    scheduler.step()
print("done!")
