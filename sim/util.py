from podm.metrics import BoundingBox
from typing import List
import os
from tqdm import tqdm

"""
BoundingBox: image, category, xmin, ymin, xmax, ymax, score(None)
GT format: frame_id, category, xmin, ymin, xmax, ymax, score, category_id 
"""


class GT():
    def __init__(self, gt_acc_path) -> None:
        self.gt_acc_path = gt_acc_path
        self.gt = self.read_gt()

    def read_gt(self):
        gt_acc_files = os.listdir(self.gt_acc_path)
        gt = {config[:-4]: [] for config in gt_acc_files}
        for gt_acc_file in tqdm(gt_acc_files, desc="gt acc"):
            config = gt_acc_file[:-4]
            with open(f"{self.gt_acc_path}/{gt_acc_file}", 'r') as f:
                acc = f.readline().split(' ')
                bbox = BoundingBox(acc[0], acc[1], float(acc[2]), float(
                    acc[3]), float(acc[4]), float(acc[5]), float(acc[6]))
                gt[config].append(bbox)
        return gt

    def show(self):
        print(self.gt["1920x1080"][100].category)


class Eval():
    def __init__(self, gt_path, iou_threshold, frames_num) -> None:
        self.gt = {}
        self.predict = None
        self.gt_path = gt_path
        self.iou_threshold = iou_threshold
        self.frames_num = frames_num
        self.init()

    def init(self):
        gt_files = os.listdir(self.gt_path)
        for gt_file in gt_files:
            config = gt_file[:-4]
            if config not in self.gt:
                self.gt[config] = [[] for i in range(self.frames_num)]
            with open(f"{self.gt_path}/{gt_file}", 'r') as f:
                parse = f.readline().split(' ')
                frame_id = int(parse[0])
                gt_item = BoundingBox.of_bbox(frame_id, parse[1], float(parse[2]), float(
                    parse[3]), float(parse[4]), float(parse[5]), float(parse[6]))
                self.gt[config][frame_id].append(gt_item)

    def test(self):
        print(self.gt["1920x1080"][0])


if __name__ == '__main__':
    eval = Eval()
