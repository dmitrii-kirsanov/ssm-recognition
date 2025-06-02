import torch
import os
import cv2
import time
from core.model.model import Model

PRETRAINED_WEIGHTS_PATH = "./data/weights"

model = Model()
model.load_state_dict(torch.load(
    os.path.join(PRETRAINED_WEIGHTS_PATH, "on_299_v_1.11_e_049_iou_0.8529.pth"),
    weights_only=True)
)
model.to("cuda")
model.training = False

markers_info = {  # sorted accordingly to top freq
    "1": "stand",
    "2": "watch (a person)",
    "3": "talk to",
    "4": "listen to (a person)",
    "5": "sit",
    "6": "carry/hold (an object)",
    "7": "walk",
    "8": "touch (an object)",
    "9": "bend/bow (at the waist)",
    "10": "lie/sleep",
}


def draw_output_on_img(cv2_image, output):
    _predictions = output.to("cpu").to(torch.float32)

    ih, iw, ic = cv2_image.shape

    for _prediction in _predictions:
        if torch.max(_prediction[4:]) < 0.65:
            continue

        cx, cy, w, h = _prediction[0:4]

        cx, cy, w, h = cx * iw, cy * ih, w * iw, h * ih

        tl = (int(cx - w / 2), int(cy - h / 2))
        br = (int(cx + w / 2), int(cy + h / 2))

        cv2_image = cv2.rectangle(cv2_image, tl, br, (0, 0, 255), 2)

        for i in range(10):
            wh = (tl[0], tl[1] + 20 * i)
            pred_val = _prediction[4 + i]
            text_val = markers_info[f"{i + 1}"]

            if pred_val > 0.65:
                cv2_image = cv2.putText(
                    cv2_image,
                    text_val, wh, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return cv2_image


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        _img = cv2.resize(img, (224, 224))

        img_tensor = torch.Tensor(_img).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).cuda()
        # print(img_tensor.shape)
        img_tensor[:, :, 0, :, :] = (img_tensor[:, :, 0, :, :] / 255.0 - 0.485) / 0.229  # R
        img_tensor[:, :, 1, :, :] = (img_tensor[:, :, 1, :, :] / 255.0 - 0.456) / 0.224  # G
        img_tensor[:, :, 2, :, :] = (img_tensor[:, :, 2, :, :] / 255.0 - 0.485) / 0.225  # B

        with torch.no_grad():
            output = model(img_tensor)
        # print(output.shape)
        img = draw_output_on_img(img, output[0, 0])

        cv2.imshow('webcam -> model', img)
        time.sleep(0.05)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_webcam(mirror=True)
