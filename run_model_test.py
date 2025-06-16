import numpy
import torch
import os
import cv2
import time
from core.model.model import Model
from core.model.ssm.ssm_block import SSM_Block

PRETRAINED_WEIGHTS_PATH = "./data/weights"

model = Model(seq_len=320, num_pred=5, num_classes=10)
# model.load_state_dict(torch.load(
#     os.path.join(PRETRAINED_WEIGHTS_PATH, "experimental_ssm_v3.00_e_014_iou_0.6620_cl_1.0921.pth"),
#     weights_only=True),
#     strict=True
# )

model.to("cuda")
model.check_stability()

model.eval()
model.bbox_classifier.ssm_block_1.set_mode(to_rnn=True, device="cuda")
model.bbox_classifier.ssm_block_2.set_mode(to_rnn=True, device="cuda")
model.bbox_detector.ssm_block_1.set_mode(to_rnn=True, device="cuda")
model.bbox_detector.ssm_block_2.set_mode(to_rnn=True, device="cuda")

As_eig = list(
    torch.linalg.eigvals(A).real.max().item() for A in model.bbox_classifier.ssm_block_1.ssm_layer.naive_repr[0])
print(max(As_eig))
#exit()


threshold = 0.65

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
        if torch.max(_prediction[4:]) < threshold:
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

            if pred_val > threshold:
                cv2_image = cv2.putText(
                    cv2_image,
                    text_val, wh, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return cv2_image


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()

        #img = numpy.random.random((480, 720, 3))
        # time.sleep(0.25)

        if mirror:
            img = cv2.flip(img, 1)

        st = time.time()
        print(torch.max(model.bbox_classifier.ssm_block_1.ssm_layer.x_state.real),
              torch.min(model.bbox_classifier.ssm_block_1.ssm_layer.x_state.real))
        # print(torch.max(model.bbox_classifier.ssm_block_2.ssm_layer.x_state.real),
        #      torch.min(model.bbox_classifier.ssm_block_2.ssm_layer.x_state.real))
        with torch.no_grad():
            output = model.inference(img)
            #print(torch.max(output[:, :2]), torch.max(output[:, 2:4]), torch.max(output[:, 4:]))
        et = time.time()

        #img = draw_output_on_img(img, output)

        cv2.imshow('webcam -> model', img)
        print(f"avg time in ms: {et - st} (output check: {torch.max(output[:, :4])} \t {torch.max(output[:, 4:])})")
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_webcam(mirror=True)
