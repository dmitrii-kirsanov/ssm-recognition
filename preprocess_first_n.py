import argparse
import os
from preprocess_video import preprocess_single_video

def preprocess_n(videos_path, csv_path, save_path, n, fps, frame_size, force):
    filenames = os.listdir(videos_path)
    i = 1

    for filename in filenames:
        path = os.path.join(videos_path, filename)
        print(f"process {filename} ({i}/{n})")
        preprocess_single_video(path, csv_path, save_path, fps, frame_size, force)

        i += 1
        if i > n:
            break
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Preprocess first n videos in dir')
    parser.add_argument("--videos_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame_size", type=str, default="224:224")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.videos_path) or not os.path.exists(args.csv_path):
        print("incorrect path: no such files")
        exit(0)

    preprocess_n(args.videos_path, args.csv_path, args.save_path, args.n, args.fps, args.frame_size, args.force)