import torch
import cv2
import argparse
import numpy as np
import os
from tqdm import tqdm


class Detector():
    def __init__(self, model_type):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.model = torch.hub.load(
            'ultralytics/yolov5', self.model_type, pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.frame_counter = 0

    def detect(self, frame, frame_num=None):
        """detect single frame and return the result.
        @params:
            frame: filename or cv2 image or pytorch image or numpy image or PIL image
        @return:
            result: processing_time, confidence, boxes
        """
        result = self.model(frame)
        result_table = result.pandas().xyxy[0].iterrows()
        process_time = sum(result.t)
        # confidence = result_table["confidence"].values.tolist()
        # x_min = result_table["xmin"].values.tolist()
        # y_min = result_table["ymin"].values.tolist()
        # x_max = result_table["xmax"].values.tolist()
        # y_max = result_table["ymax"].values.tolist()
        # object_name = result_table["name"].values.tolist()
        self.frame_counter += 1
        detect_result = []
        for _, item in result_table:
            item = item.values.tolist()
            item.insert(0, process_time)
            if frame_num is not None:
                item.insert(0, f"{frame_num:06d}")
            detect_result.append(item)
        # [[(frame_num), process_time, xmin, ymin, xmax, ymax, confidence, class_id, name]]
        return detect_result

    def save(self, result, f):
        frame_num = result["frame_num"]
        bytes_in_size = result["size"]
        accuracy = result["accuracy"]
        process_time = result["process_time"]
        f.write(frame_num, " ", bytes_in_size, " ",
                process_time, " ", accuracy, "\n")

    def reset(self):
        self.frame_counter = 0

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

    def test(self, video_file):
        cap = cv2.VideoCapture(video_file)
        print("test")
        while (cap.isOpened()):
            _, frame = cap.read()
            if frame is not None:
                self.detect(frame)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_single_frame(self, video_path):
        _, _, files = next(os.walk(video_path))
        crf_config = []
        for video_file in tqdm(files, desc="processing"):
            config = video_file[:-4].split('_')
            width = int(config[0])
            height = int(config[1])
            # frame_rate = int(config[2])
            constant_rate_factor = int(config[3])
            frame_counter = 0
            config = [width, height, constant_rate_factor]
            if config not in crf_config:
                crf_config.append(config)
                print(
                    f"split and saving file in config: {config[0]}:{config[1]}:{config[2]}")
                path = video_path+'/'+f"{config[0]}_{config[1]}_{config[2]}"
                if not os.path.exists(path):
                    print(f"making new folder {path}")
                    os.makedirs(path)
                cap = cv2.VideoCapture(video_path + "/" + video_file)
                while(cap.isOpened()):
                    _, frame = cap.read()
                    if frame is not None:
                        frame_counter += 1
                        cv2.imwrite(
                            f"{path}/{frame_counter:06d}.jpg", frame)
                    else:
                        break
                cap.release()

    def pre_detect(self, path, saving_path):
        """preprocess to generate a detection table for each configuration.
        @params:
            path: path that contains all processed seperated images
            saving_path: path that used for store the pre-detection result
        """
        print(f"path {path}")
        _, dirs, _ = next(os.walk(path))
        for dir in tqdm(dirs, desc="processing"):
            self.reset()
            if not os.path.exists(f"{saving_path}/{dir}"):
                os.makedirs(f"{saving_path}/{dir}")
                print(f"makeing new folder: {saving_path}/{dir}")
            with open(f"{saving_path}/{dir}/{dir}_{self.model_type}.csv", 'w') as f:
                print(f"pre-detection: {dir}")
                for frame_name in sorted(os.listdir(f"{path}/{dir}")):
                    frame_path = f"{path}/{dir}/{frame_name}"
                    frame_num = int(frame_name[:-4])
                    result = self.detect(frame_path, frame_num)
                    file_size = os.path.getsize(frame_path)
                    for item in result:
                        item.insert(1, file_size)
                        f.write(' '.join([str(v) for v in item]))
                        f.write('\n')


if __name__ == "__main__":
    # run: python pre_detect.py --filepath=test_data/test.flv
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str,
                        help="yolov5n, yolov5s, yolov5m, yolov5l, yolov5x")
    parser.add_argument("--filepath", type=str, help="test video file path")
    parser.add_argument("--video_path", type=str,
                        help="videos path for save single images")
    parser.add_argument("--saving_path", type=str,
                        help="pre-detection csv saving path")
    args = parser.parse_args()
    if args.model_type:
        if args.filepath:
            detector = Detector(args.model_type)
            detector.test(args.filepath)
        if args.video_path and args.saving_path is None:
            detector = Detector(args.model_type)
            detector.save_single_frame(args.video_path)
        if args.saving_path and args.video_path:
            detector = Detector(args.model_type)
            detector.pre_detect(args.video_path, args.saving_path)
