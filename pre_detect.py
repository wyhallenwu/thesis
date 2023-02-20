import torch
import numpy as np


class Detector():
    def __init__(self, path, model_type):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.path = path
        self.model = torch.hub.load(
            'ultralytics/yolov5', model_type, pretrained=True, trust_repo=True)
        self.model.to(self.device)

    def detect(self, frame):
        result = self.model(frame)
        result_table = result.pandas().xyxy[0]
        process_time = sum(result.t)
        confidence = result_table["confidence"]
        x_min = result_table["xmin"]
        y_min = result_table["ymin"]
        x_max = result_table["xmax"]
        y_max = result_table["ymax"]
        return process_time, confidence, [x_min, y_min, x_max, y_max]

    def save(self, result, f):
        frame_num = result["frame_num"]
        bytes_in_size = result["size"]
        accuracy = result["accuracy"]
        process_time = result["process_time"]
        f.write(frame_num, " ", bytes_in_size, " ",
                process_time, " ", accuracy, "\n")

    def mAP(self):
        """compute mean average precision(mAP).
        """
        pass

    def compute_iou(raw_box, gt_box):
        """compute iou of the raw and ground truth box.
        @params:
            raw_box, gt_box: [x_min, y_min, x_max, y_max]
        @return:
            iou
        """
        # compute corner cordinates
        inter_area_left_lower_x = np.max([raw_box[0], gt_box[0]])
        inter_area_left_lower_y = np.max([raw_box[1], gt_box[1]])
        inter_area_right_upper_x = np.min([raw_box[2], gt_box[2]])
        inter_area_right_upper_y = np.min([raw_box[3], gt_box[3]])
        # computer inter area
        raw_area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        inter_area = (np.max([0, inter_area_right_upper_x-inter_area_left_lower_x])) * (
            np.max([0, inter_area_right_upper_y-inter_area_left_lower_y]))
        # compute iou
        iou = inter_area / (raw_area + gt_area - inter_area)
        return iou

    def compare_gt(self, result, gt):
        """compare detection result with ground truth and record accuracy
        @params:
            result: detection result
            gt: ground truth
        @return:
            accuracy
        """
        pass
