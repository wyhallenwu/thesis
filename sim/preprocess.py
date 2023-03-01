import os
import cv2
from tqdm import tqdm
import argparse
from detectors import YoloDetector, DetrDetector
CONFIG = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]


def resize(src_frame_path, saving_path):
    assert os.path.exists(src_frame_path)
    frames = sortedos.listdir(src_frame_path)
    for config in tqdm(CONFIG, desc="resizing"):
        if not os.path.exists(f"{saving_path}/{config[0]}x{config[1]}"):
            os.makedirs(f"{saving_path}/{config[0]}x{config[1]}")
            print(f"making new folder {saving_path}/{config[0]}x{config[1]}")
        for frame in sorted(frames):
            img = cv2.imread(f"{src_frame_path}/{frame}")
            resized_img = cv2.resize(
                img, config, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(
                f"{saving_path}/{config[0]}x{config[1]}/{frame}", resized_img)


def generate_gt(detector, gt_path, saving_path):
    assert os.path.exists(saving_path)
    _, dirs, _ = next(os.walk(gt_path))
    for dir in tqdm(dirs, desc="dir"):
        frames = sorted(os.listdir(f"{gt_path}/{dir}"))
        if not os.path.exists(f"{saving_path}/{detector.model_type}"):
            os.makedirs(f"{saving_path}/{detector.model_type}")
        with open(f"{saving_path}/{detector.model_type}/{dir}.csv", 'w') as f:
            for frame in frames:
                frame_id = frame.split('.')[0]
                result = detector.detect(f"{gt_path}/{dir}/{frame}", frame_id)
                for item in result:
                    f.write(' '.join(list(map(lambda x: str(x), item))))
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        help="src frames dataset path. such MOT16-04")
    parser.add_argument("--saving", type=str, help="saving resized image path")
    parser.add_argument("--gt_path", type=str,
                        help="resized frames ground truth path")
    parser.add_argument(
        "--model", type=str, help="detection model type. yolov5n, yolov5m, yolov5l, yolov5x, detr")
    args = parser.parse_args()
    if args.src and args.saving:
        resize(args.src, args.saving)
    if args.gt_path and args.saving and args.model:
        if args.model[:4] == "yolo":
            detector = YoloDetector(args.model)
        if args.model == "detr":
            detector = DetrDetector()
        generate_gt(detector, args.gt_path, args.saving)
