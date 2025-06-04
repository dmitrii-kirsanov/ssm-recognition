import torch
import tqdm

from core.loss.loss import loss_v2
from core.model.model import Model
from core.util.custom_dataset import CustomDataset

torch.manual_seed(0)
torch.set_float32_matmul_precision('high')  # warning recommend to do it

criterion, batch_size = loss_v2, 60
num_epochs, num_videos = 100, 29
seconds_seq_len, fps = 1, 1
classes_num, pred_num = 10, 5

model = Model(seq_len=seconds_seq_len*fps, num_pred=pred_num, num_classes=classes_num)
#model.load_pretrain_backbone("./data/pretrain/v8_m.pth")
model.load("final_non_ssm_v_1.11_e_060_iou_0.8754_class_loss_0.0452.pth")
model.freeze_backbone(True)
model.to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.amp.GradScaler('cuda')

cd = CustomDataset(preprocessed_directory="./data/ava/preprocessed_videos/", fps=fps, seconds_seq_len=seconds_seq_len,
                   max_videos=num_videos, classes_num=classes_num, pred_max=pred_num, dest_device=torch.device("cuda"))
dataloader = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch_id in range(num_epochs):
    running_iou, running_class_loss = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    pbar = tqdm.tqdm(dataloader, desc="epoch training")
    for batch in pbar:
        images, targets = batch  # images [B, SL, 3, H, W], targets [B, SL, num_pred, 4(bb) + 10(classes)]

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            b, sl, n, po = outputs.shape
            _, _, m, pt = targets.shape
            loss, avg_target_iou, class_loss = criterion(outputs.reshape(b * sl, n, po), targets.reshape(b * sl, m, pt),
                                                         num_classes=10,
                                                         k_bbox_loss=0.8, k_class_loss=0.2,
                                                         threshold_iou_for_class_loss=0.75)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_iou += avg_target_iou
        running_class_loss += class_loss

    avg_iou = running_iou.item() / len(dataloader)
    avg_class_loss = running_class_loss.item() / len(dataloader)

    print(f"epoch: {epoch_id:03d}, avg_target_iou: {avg_iou}, avg_class_loss: {avg_class_loss}")
    model.save(f"experimental_ssm_v_0.02_e_{epoch_id:03d}_iou_{avg_iou:4.4f}_cl_{avg_class_loss:4.4f}.pth")

    scheduler.step()
print("done!")
