import torch
import numpy as np
import cv2

conf_threshold = 0.75


def show_new_img_bbox(img, targets, predictions, class_check = False, save_path=None):
    with torch.no_grad():
        _img = img.to("cpu").to(torch.float32)
        _predictions = predictions.to("cpu").to(torch.float32)
        _targets = targets.to("cpu").to(torch.float32)

        numpy_image = _img.numpy()

        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2_image.astype(np.uint8)
        cv2_image = cv2_image.copy()

        cv2_image = cv2.resize(cv2_image, (0, 0), fx=4.5, fy=4.5)

        ih, iw, ic = cv2_image.shape

        for _prediction in _predictions:
            cx, cy, w, h = _prediction[0:4]

            if torch.all(_prediction[4:] < conf_threshold) and class_check:
                continue

            cx, cy, w, h = cx * iw, cy * ih, w * iw, h * ih

            tl = (int(cx - w / 2), int(cy - h / 2))
            br = (int(cx + w / 2), int(cy + h / 2))

            cv2_image = cv2.rectangle(cv2_image, tl, br, (0, 0, 255), 2)
            cv2_image = cv2.putText(
                cv2_image,
                f"{_prediction[4:]}", (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1, cv2.LINE_AA)


        for _target in _targets:
            cx, cy, w, h = _target[0:4]

            cx, cy, w, h = cx * iw, cy * ih, w * iw, h * ih

            tl = (int(cx - w / 2), int(cy - h / 2))
            br = (int(cx + w / 2), int(cy + h / 2))

            cv2_image = cv2.rectangle(cv2_image, tl, br, (255, 0, 0), 2)
            cv2_image = cv2.putText(
                cv2_image,
                f"{_target[4:]}", (tl[0], br[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1, cv2.LINE_AA)

        if save_path:
            cv2.imwrite(save_path, cv2_image)
            return

        cv2.imshow("Image", cv2_image)
        cv2.waitKey(0)