import torch
from core.util.custom_dataset import CustomDataset
from core.util.draw_output import show_new_img_bbox

DATA_PATH = "./data/ava/preprocessed_videos/"
batch_size = 1
seconds_seq_len = 10

if __name__ == "__main__":
    # a = torch.randn(10).reshape(10, 1) #строка
    # b = torch.randn(10) # столбец
    # c = a * b
    # print(c.shape)

    cd = CustomDataset(DATA_PATH, fps=1, seconds_seq_len=seconds_seq_len, max_videos=5, classes_num=10, pred_max=5)
    dataloader = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=True,
                                             num_workers=0)

    for batch in dataloader:
        images, targets = batch
        show_new_img_bbox(images[0, 0], targets[0, 0], torch.zeros(0))
        #print(targets)

