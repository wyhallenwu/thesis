from podm.metrics import BoundingBox, get_pascal_voc_metrics, MetricPerClass
from typing import List
import os
from tqdm import tqdm


# BoundingBox: image, category, xmin, ymin, xmax, ymax, score(None)
# GT format: frame_id, category, xmin, ymin, xmax, ymax, score


class GT():
    """GT is the collection of the pre-analyzed groundtruth."""

    def __init__(self, gt_acc_path, model_type, gt_frames_num) -> None:
        self.gt_acc_path = gt_acc_path
        self.model_type = model_type
        self.model_gt = "detr" if self.model_type == "detr" else "yolov5x"
        self.gt_frames_num = gt_frames_num
        self.gt = self.init_gt()  # gt: Dict[config: List[List[BoundingBox]]

    def init_gt(self):
        """load groundtruth of the corresponding detection model."""
        gt_acc_files = os.listdir(f"{self.gt_acc_path}/{self.model_gt}")
        gt = {file[:-4]: [[]
                          for _ in range(self.gt_frames_num)] for file in gt_acc_files}
        for gt_acc_file in tqdm(gt_acc_files, desc="gt acc"):
            config = gt_acc_file[:-4]
            with open(f"{self.gt_acc_path}/{self.model_gt}/{gt_acc_file}", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    acc = line.strip().split(' ')
                    bbox = BoundingBox.of_bbox(acc[0], acc[1], float(acc[2]), float(
                        acc[3]), float(acc[4]), float(acc[5]), float(acc[6]))
                    gt[config][int(acc[0]) - 1].append(bbox)
        return gt

    def get_boundingboxes(self, resolution, frame_id):
        """return the groundtruth boundingbox within the corresponding resolution and frame_id.
        @params:
            resolution: [width, height]
            frame_id(int): index of frame
        """
        resolution = f"{resolution[0]}x{resolution[1]}"
        return self.gt[resolution][frame_id - 1]

    def test(self):
        print("ground truth frames num: ", self.gt_frames_num)
        print("ground truth configs: ", self.gt.keys())
        print("ground truth bbox: ", next(iter(self.gt.values()))[0])
        print("ground truth bboxes")
        for box in next(iter(self.gt.values()))[0]:
            print(box.category, box.score)


class Evaluator():
    """Evaluator evaluates the predition with the groundtruth."""

    def __init__(self, gt_acc_path, model_type, frames_num, iou_threshold=0.5) -> None:
        self.model_type = model_type
        self.gt_acc_path = gt_acc_path
        self.gt = GT(gt_acc_path, self.model_type, frames_num)
        self.iou_threshold = iou_threshold
        self.frames_num = frames_num

    def evaluate(self, prediction, resolution, frame_id):
        """evaluate the prediction of a frame with the corresponding config groudtruth.
        @params:
            prediction(List[BoundingBox]): detected boundingboxes in the frame
            resolution: [width, height]
            frame_id(int): the frame index in the original stream
        @returns:
            ret(Dict): {presion, recall, ap, tp, fp} of a single frame
            mAp(float): mAp
        """
        ret = {}
        result = get_pascal_voc_metrics(
            self.gt.get_boundingboxes(resolution, frame_id), prediction)
        for k, v in result.items():
            ret[k] = {"precision": v.precision, "recall": v.recall, "ap": v.ap, "tp": v.tp,
                      "fp": v.fp}
        return ret, MetricPerClass.mAP(result)


def energy_consuming(frames_num, resolution, local=False):
    """energy consumed to process n frames.
    following the setting of paper: Joint Configuration Adaptation and Bandwidth Allocation for Edge-based Real-time Video Analytics
    @params:
        frames_num(int): frames the segment contained
        resolution(List[int]): [width, height]
        local(bool): if True the segment is processed by local client else by the remote server
    """
    mu = 5  # 5j/frame
    gamma = 5e-6
    alpha = 1
    width, height = resolution
    processing_energy = mu * frames_num if local else 0
    transmission_energy = alpha * gamma * \
        frames_num * (width * height * 8) ** 2 if not local else 0
    return processing_energy, transmission_energy


if __name__ == '__main__':
    gt_eval = Evaluator("acc", "detr", 1050)
    gt_eval.gt.test()
