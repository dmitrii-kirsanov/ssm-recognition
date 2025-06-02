import argparse
import os
import wget

def load_ava_dataset(path, load_train, load_test, size_limit):
    os.makedirs(path, exist_ok=True)

    # load markers
    if not os.path.exists(f"{path}/ava_v2.2.zip"):
        wget.download("https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip", out=path)
    if not os.path.exists(f"{path}/markers"):
        os.system(f"unzip -j {path}/ava_v2.2.zip -d {path}/markers")

    if load_train:
        # load videos names
        if not os.path.exists(f"{path}/ava_file_names_trainval_v2.1.txt"):
            wget.download("https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt", out=path)

        # load videos
        os.makedirs(f"{path}/train_videos", exist_ok=True)
        i_stop = size_limit
        with open(f"{path}/ava_file_names_trainval_v2.1.txt", "r") as trainlist_file:
            for line in trainlist_file.readlines():
                line = line[:-1] #remove \n

                if not os.path.exists(f"{path}/train_videos/{line}"):
                    wget.download(f"https://s3.amazonaws.com/ava-dataset/trainval/{line}", out=f"{path}/train_videos/")
                print(f"({size_limit - i_stop + 1}/{size_limit} video done)")

                i_stop -= 1
                if i_stop <= 0:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Dataset download script')
    parser.add_argument("name", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true") #TODO: impl
    parser.add_argument("--size_limit", type=int, default=2)

    args = parser.parse_args()
    if args.name == "ava":
        load_ava_dataset(args.path, args.train, args.test, args.size_limit)