import torch
import tqdm

from core.loss.loss import loss_v2
from core.model.model import Model
from core.util.custom_dataset import CustomDataset

torch.manual_seed(0)
torch.set_float32_matmul_precision('high')  # warning recommend to do it

version_name = "v3.00"

criterion, batch_size = loss_v2, 1
num_epochs, num_videos = 100, 2#299
seconds_seq_len, fps = 80, 4
# criterion, batch_size = loss_v2, 60
# num_epochs, num_videos = 100, 299
# seconds_seq_len, fps = 1, 1
startup_epoch = 0

classes_num, pred_num = 10, 5

model = Model(seq_len=seconds_seq_len*fps, num_pred=pred_num, num_classes=classes_num)
model.load_pretrain_backbone("./data/pretrain/v8_m.pth")
#model.load("experimental_ssm_v2.20_clear_ssm_blocks.pth", strict=False)
model.freeze_backbone(True)
model.freeze_classifier(True)
model.to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.amp.GradScaler('cuda')

for _ in range(startup_epoch):
    scheduler.step()

cd = CustomDataset(preprocessed_directory="./data/ava/preprocessed_videos/", fps=fps, seconds_seq_len=seconds_seq_len,
                   max_videos=num_videos, classes_num=classes_num, pred_max=pred_num, dest_device=torch.device("cuda"))
dataloader = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch_id in range(num_epochs):
    running_iou, running_class_loss = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    running_prec, running_rec = torch.zeros(classes_num, device="cuda"), torch.zeros(classes_num, device="cuda")

    pbar = tqdm.tqdm(dataloader, desc=f"epoch training ({epoch_id:03d})")
    for batch in pbar:
        images, targets = batch  # images [B, SL, 3, H, W], targets [B, SL, num_pred, 4(bb) + 10(classes)]

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            b, sl, n, po = outputs.shape
            _, _, m, pt = targets.shape
            loss, avg_target_iou, class_loss, precision, recall = criterion(outputs.reshape(b * sl, n, po), targets.reshape(b * sl, m, pt),
                                                         num_classes=10,
                                                         k_bbox_loss=1.0, k_class_loss=0.0,
                                                         threshold_iou_for_class_loss=1.0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_iou += avg_target_iou
        running_class_loss += class_loss
        running_prec += precision
        running_rec += recall

    avg_iou = running_iou.item() / len(dataloader)
    avg_class_loss = running_class_loss.item() / len(dataloader)
    avg_prec = running_prec / len(dataloader)
    avg_rec = running_rec / len(dataloader)

    print(f"avg_target_iou: {avg_iou}, avg_class_loss: {avg_class_loss}")
    print(f"avg_prec: { list(f'{number:2.4f}' for number in avg_prec.cpu().numpy()) }")
    print(f"avg_rec: { list(f'{number:2.4f}' for number in avg_rec.cpu().numpy()) }")

    model.save(f"experimental_ssm_{version_name}_e_{epoch_id:03d}_iou_{avg_iou:4.4f}_cl_{avg_class_loss:4.4f}.pth")
    scheduler.step()

    torch.save(optimizer.state_dict(), f"./data/train_tools_states/optim_{version_name}_{epoch_id:03d}.pth")
    torch.save(scheduler.state_dict(), f"./data/train_tools_states/shed_{version_name}_{epoch_id:03d}.pth")

print("done!")
