import argparse
import os
import csv
import json
import shutil

#fps = 10

def preprocess_single_video(video_path, csv_path, save_path, fps, frame_size, force):
    video_name = video_path.split('/')[-1].split('.')[0]
    folder_path = os.path.join(save_path, video_name)
    timestamps = {}

    if os.path.exists(os.path.join(save_path, video_name)) and not force:
        print(f"already preprocessed (destination path existed, force = {force})")
        return

    #read bboxes from csv
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in reader:
            name, sec, bbox_x1, bbox_x2, bbox_y1, bbox_y2, action_id, person_id = row[0].split(',')
            sec = str(int(sec)) # remove first zeros

            if name == video_name:
                curr_bbox = {
                        "bbox": [bbox_x1, bbox_x2, bbox_y1, bbox_y2],
                        "action_id": action_id,
                        "person_id": person_id
                }
                if not sec in timestamps:
                    timestamps[sec] = []
                timestamps[sec].append(curr_bbox)

    if not len(timestamps.keys()): # maybe it just in "test/val", not in "train"
        print(f"INVALID INTER MARKERS, no markers in csv file ({video_name}). stop preprocessing file...")
        return

    #save bboxes info
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/timestamps.json', 'w') as fp:
        json.dump(timestamps, fp, sort_keys=True, indent=4, separators=(',', ': '))

    #dir for ffmpeg
    rawimages_path = os.path.join(folder_path, "raw_images")
    os.makedirs(rawimages_path, exist_ok=True)

    #get time and duration of cutting video part, cut video -> images
    min_sec = min(int(k) for k in timestamps.keys())
    max_sec = max(int(k) for k in timestamps.keys())
    sec_len = max_sec - min_sec + 1
    print(f"{min_sec=}, {max_sec=}, {sec_len=}")
    s_h, s_m, s_s = int(min_sec / 3600), int((min_sec / 60) % 60), int(min_sec % 60)
    t_h, t_m, t_s = int(sec_len / 3600), int((sec_len / 60) % 60), int(sec_len % 60)
    print("calling ffmpeg...")
    os.system(f"""ffmpeg -hide_banner -loglevel error -i {video_path} -ss {s_h}:{s_m}:{s_s} -t {t_h}:{t_m}:{t_s} -vf "fps={fps},scale={frame_size}" "{rawimages_path}/rawindex_%d.png" """)

    assert sec_len * fps == len(os.listdir(rawimages_path))  # check if images_count = seconds * fps

    #restructure images
    images_path = os.path.join(folder_path, "images")
    os.makedirs(images_path, exist_ok=True)
    for i in range(sec_len):
        sec = min_sec + i
        for j in range(fps):
            shutil.copyfile(os.path.join(rawimages_path, f"rawindex_{i * fps + j + 1}.png"), os.path.join(images_path, f"{sec}_{j}.png"))

    assert sec_len * fps == len(os.listdir(images_path))  # double check, ~//~
    shutil.rmtree(rawimages_path)  # remove tmp dir for ffmpeg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Single video preprocess script')
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frame_size", type=str, default="640:480")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.video_path) or not os.path.exists(args.csv_path):
        print("incorrect path: no such files")
        exit(0)

    preprocess_single_video(args.video_path, args.csv_path, args.save_path, args.fps, args.frame_size, arg.force)