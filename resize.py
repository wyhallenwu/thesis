import cv2
import argparse
import os
from tqdm import tqdm

resolution = [[1920, 1080], [1600, 900], [1280, 720], [960, 540], [640, 360]]


def resize(path, rs, saving_path):
    files = sorted(os.listdir(path))
    for file in tqdm(files, desc=f"{rs[0]}x{rs[1]}"):
        frame_num = file[:-4]
        frame = cv2.imread(path + '/' + file)
        resized_frame = cv2.resize(frame, rs, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(saving_path + '/' + frame_num + '.bmp', resized_frame)


def main(frame_path, saving_path):
    for rs in resolution:
        if not os.path.exists(f"{saving_path}/{rs[0]}_{rs[1]}"):
            os.makedirs(f"{saving_path}/{rs[0]}_{rs[1]}")
            print(f"making new folder {saving_path}/{rs[0]}_{rs[1]}")
        resize(frame_path, rs, f"{saving_path}/{rs[0]}_{rs[1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, help="original frames")
    parser.add_argument("--saving_path", type=str, help="saving result path")
    args = parser.parse_args()
    if args.frame_path and args.saving_path:
        main(args.frame_path, args.saving_path)
