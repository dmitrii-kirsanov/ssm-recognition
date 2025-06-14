import torch
import tqdm

from core.loss.loss import loss_v2
from core.model.model import Model
from core.util.custom_dataset import CustomDataset

torch.manual_seed(0)
torch.set_float32_matmul_precision('high')  # warning recommend to do it

criterion, batch_size = loss_v2, 60
num_videos = 299
seconds_seq_len, fps = 1, 1

classes_num, pred_num = 10, 5

model = Model(seq_len=seconds_seq_len*fps, num_pred=pred_num, num_classes=classes_num)
model.load("final_non_ssm_v_1.11_e_060_iou_0.8754_class_loss_0.0452.pth")
model.to("cuda")

cd = CustomDataset(preprocessed_directory="./data/ava/preprocessed_videos/", fps=fps, seconds_seq_len=seconds_seq_len,
                   max_videos=num_videos, classes_num=classes_num, pred_max=pred_num, dest_device=torch.device("cuda"))
dataloader = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch_id in range(1):
    running_iou, running_class_loss = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    #running_prec, running_rec = torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda")
    running_prec, running_rec = torch.zeros(classes_num, device="cuda"), torch.zeros(classes_num, device="cuda")
    pbar = tqdm.tqdm(dataloader, desc="calculating")
    for batch in pbar:
        images, targets = batch  # images [B, SL, 3, H, W], targets [B, SL, num_pred, 4(bb) + 10(classes)]

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                outputs = model(images)
            b, sl, n, po = outputs.shape
            _, _, m, pt = targets.shape
            loss, avg_target_iou, class_loss, precision, recall = criterion(outputs.reshape(b * sl, n, po), targets.reshape(b * sl, m, pt),
                                                         num_classes=10,
                                                         k_bbox_loss=0.8, k_class_loss=0.2,
                                                         threshold_iou_for_class_loss=0.75)

        running_iou += avg_target_iou
        running_class_loss += class_loss
        running_prec += precision
        running_rec += recall

    # avg_iou = running_iou.item() / len(dataloader)
    # avg_class_loss = running_class_loss.item() / len(dataloader)
    # avg_prec = running_prec.item() / len(dataloader)
    # avg_rec = running_rec.item() / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    avg_class_loss = running_class_loss / len(dataloader)
    avg_prec = running_prec / len(dataloader)
    avg_rec = running_rec / len(dataloader)

    print(f"avg_target_iou: {avg_iou}, avg_class_loss: {avg_class_loss}")
    print(f"avg_prec: {avg_prec}")
    print(f"avg_rec: {avg_rec}")
print("done!")
