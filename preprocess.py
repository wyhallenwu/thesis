import os
import cv2
from tqdm import tqdm
CONFIG = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]


def resize(src_frame_path, saving_path):
    frames = os.listdir(src_frame_path)
    for config in tqdm(CONFIG, desc=f"{config}"):
        if not os.path.exists(f"{saving_path}/{config}"):
            os.makedirs(f"{saving_path}/{config}")
            print(f"making new folder {saving_path}/{config}")
        for frame in frames:
            img = cv2.imread(f"{src_frame_path}/{frame}")
            resized_img = cv2.resize(
                img, config, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{saving_path}/{config}/{frame}", resized_img)
