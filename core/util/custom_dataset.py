import os
import json
import tqdm
import torch
from torchvision.ops import box_convert

import cv2

rematch_markers = {  # sorted accordingly to top freq
    "12": "1",  # stand
    "80": "2",  # watch (a person)
    "79": "3",  # talk to
    "74": "4",  # listen to (a person)
    "11": "5",  # sit
    "17": "6",  # carry/hold (an object)
    "14": "7",  # walk
    "59": "8",  # touch (an object)
    "1": "9",  # bend/bow (at the waist)
    "8": "10",  # lie/sleep
}


def rematch_action_indices(gt_json):
    new_gt_json = []
    for action in gt_json:
        if not action["action_id"] in rematch_markers:
            continue

        action["action_id"] = rematch_markers[action["action_id"]]
        new_gt_json.append(action)
    return new_gt_json


def rematch_bbox(bbox):
    # top-left (x1, y1) and bottom-right (x2,y2) -> centre, width and height
    new_bbox = box_convert(bbox, "xyxy", "cxcywh")
    return new_bbox


def merge_same_person_id_bboxes(gt_json_sec):
    res_json = {}

    for action in gt_json_sec:
        person_id = action["person_id"]
        if not person_id in res_json:
            res_json[person_id] = action.copy()
            res_json[person_id]["action_id"] = [res_json[person_id]["action_id"]]
        else:
            res_json[person_id]["action_id"].append(action["action_id"])

    return list(res_json.values())


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, preprocessed_directory, fps, seconds_seq_len, max_videos, classes_num, pred_max, dest_device):
        assert classes_num == 10  # top 10 implement only (see rematch_action_indices)
        self.mem = []
        self.device = dest_device

        self.fps, self.seconds_seq_len = fps, seconds_seq_len
        self.dir = preprocessed_directory
        self.classes_num, self.pred_max = classes_num, pred_max
        self.videos_paths = os.listdir(self.dir)
        self.videos_paths = self.videos_paths[:min(max_videos, len(self.videos_paths))]

        for video_path in self.videos_paths:
            self.load_single_video_to_memory(video_path)

    def print_info(self):
        pass  # some statistics: every action type count, avg actions on image

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, idx):
        images, labels = self.mem[idx]
        images = [torch.Tensor(cv2.imread(image)).permute(2, 0, 1) for image in images]
        images = torch.stack(images, dim=0)

        images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

        images[:, 0, :, :] = (images[:, 0, :, :] / 255.0 - 0.485) / 0.229  # R
        images[:, 1, :, :] = (images[:, 1, :, :] / 255.0 - 0.456) / 0.224  # G
        images[:, 2, :, :] = (images[:, 2, :, :] / 255.0 - 0.485) / 0.225  # B

        return images, labels

    def form_seq_instance(self, start_seq_sec, end_seq_sec, curr_dir, timestamps):
        instance_images = []
        instance_labels = []

        for sec in range(start_seq_sec, end_seq_sec):
            images = [os.path.join(curr_dir, "images", f"{sec}_{i}.png") for i in range(self.fps)]
            instance_images.extend(images)

            gt_tensor = torch.zeros(self.pred_max, 4 + self.classes_num)  # no confidence
            gt_json = timestamps[f"{sec}"] if f"{sec}" in timestamps else []

            # rematch classes for easier use (and by the way filter them)
            gt_json = rematch_action_indices(gt_json)

            # merge bboxes by person_ids (now it's key). action_id's now is a list
            _gt_json = merge_same_person_id_bboxes(gt_json)

            # cut to self.pred_max
            gt_json = _gt_json[:self.pred_max]


            for i, obj in enumerate(gt_json):
                gt_tensor[i][0:4] = rematch_bbox(torch.Tensor([float(x) for x in obj["bbox"]]))  # bboxes
                #gt_tensor[i][3 + 1] = 1
                #gt_tensor[i][3 + int(obj["action_id"])] = 1
                for action_id in obj["action_id"]:  #
                    gt_tensor[i][3 + int(action_id)] = 1  # class # now its list of action_id (due to person_id's merge)

            gt_tensor = gt_tensor.repeat(self.fps, 1, 1)

            instance_labels.extend(gt_tensor)

        # return torch.stack(instance_images, dim = 0), torch.stack(instance_labels, dim=0)
        return instance_images, torch.stack(instance_labels, dim=0)

    def load_single_video_to_memory(self, video_name):
        curr_dir = os.path.join(self.dir, video_name)

        with open(os.path.join(curr_dir, 'timestamps.json')) as json_file:
            timestamps = json.load(json_file)

        min_sec = min(int(k) for k in timestamps.keys())
        max_sec = max(int(k) for k in timestamps.keys())

        # run through selected timeline rwith self.seconds_seq_len step
        for sec_st in tqdm.tqdm(range(min_sec, max_sec - self.seconds_seq_len, self.seconds_seq_len),
                                desc=f"load {video_name}"):
            sec_end = sec_st + self.seconds_seq_len

            instance_imgs, instance_labels = self.form_seq_instance(sec_st, sec_end, curr_dir, timestamps)
            self.mem.append((instance_imgs, instance_labels))
