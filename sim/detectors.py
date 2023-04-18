import torch
import cv2
from podm.metrics import BoundingBox
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
from sim.util import Evaluator
from abc import ABCMeta, abstractmethod


class Detector(metaclass=ABCMeta):
    """interface of any Detector"""
    @abstractmethod
    def analyze_single_frame(self, frame, frame_id, stream):
        pass

    @abstractmethod
    def analyze_single_segment(self, filename, frames_id):
        pass

    @abstractmethod
    def prediction2bbox(self, prediction):
        """convert detection result to the List of BoundingBox."""
        pass


class YoloDetector(Detector):
    """Yolo Object Detection Model.\
        https://pytorch.org/hub/ultralytics_yolov5/"""

    def __init__(self, model_type):
        """
        @params:
            model_type: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.model = torch.hub.load(
            'ultralytics/yolov5', self.model_type, pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.frame_counter = 0
        print("BUILD YOLO DETECTOR DONE.")

    def analyze_single_frame(self, frame, frame_id, stream=True):
        """analyze single frame and return the result.
        @params:
            frame: filename
            frame_id: int
            stream(bool): true if the frame captured from cv2 VideoCapture stream

        @return:
            detection_result: List[[frame_id, class, xmin, ymin, xmax, ymax, score]]
            process_time: ms
        """
        result = self.model(frame)
        result_table = result.pandas().xyxy[0]
        process_time = sum(result.t)
        confidence = result_table["confidence"].values.tolist()
        x_min = result_table["xmin"].values.tolist()
        y_min = result_table["ymin"].values.tolist()
        x_max = result_table["xmax"].values.tolist()
        y_max = result_table["ymax"].values.tolist()
        class_name = result_table["name"].values.tolist()
        detection_result = [[f"{frame_id:06d}", class_name[i].replace(' ', '_'), round(x_min[i], 3), round(y_min[i], 3),
                             round(x_max[i], 3), round(y_max[i], 3), round(
                                 confidence[i], 3), round(process_time, 3)] for i in range(len(confidence))]
        return detection_result, round(process_time, 3)

    def analyze_single_segment(self, filename, frames_id):
        """analyze single video segment and return the bboxes of each frames and processing time.
        @params:
            filename(str): video segment filename
            frames_id(List[int]): index of each frame in the ground truth order

        @return:
            bboxes(List(BoundingBox)): BoundingBox of each frame
            processing_time(float): time to analyze the video segment
        """
        cap = cv2.VideoCapture(filename)
        frames = []
        bboxes = []
        processing_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        assert len(frames) == len(frames_id), "video frames lost."
        for frame, frame_id in zip(frames, frames_id):
            result, process_time = self.analyze_single_frame(frame, frame_id)
            bboxes.append(self.prediction2bbox(result))
            processing_time += process_time
        return bboxes, processing_time

    def reset(self):
        self.frame_counter = 0

    def prediction2bbox(self, detection):
        """helper function to convert the List of detected objects' coordinates to BoundingBox"""
        bboxes = []
        for item in detection:
            bbox = BoundingBox.of_bbox(item[0], item[1], item[2],
                                       item[3], item[4], item[5], item[6])
            bboxes.append(bbox)
        return bboxes


class DetrDetector(Detector):
    """Detr Object Detection Model.\
        https://huggingface.co/facebook/detr-resnet-101"""

    def __init__(self, threshold=0.8) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101").to(self.device)
        self.threshold = threshold
        self.model_type = "detr"
        print("BUILD DETR DETECTOR DONE.")

    def analyze_single_frame(self, frame, frame_id, stream=False):
        """analyze a single frame.
        @params:
            frame(str): the filename of the image
            frame_id(int): the index of the frame starting from 1 in the format :06d
            stream(bool): true if the frame captured from cv2 VideoCapture stream
        @return:
            bboxes(List(BoundingBox)): BoundingBox of each frame
            processing_time(float): time to analyze the video segment
        """
        if not stream:
            frame = Image.open(frame)
        else:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=frame, return_tensors="pt").to(self.device)
        start_time = time.time()
        outputs = self.model(**inputs)
        end_time = time.time()
        target_sizes = torch.tensor([frame.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)[0]
        process_time = round((end_time - start_time) * 1000, 3)
        analyze_result = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 3) for i in box.tolist()]
            class_name = str(
                self.model.config.id2label[label.item()]).replace(' ', '_')
            analyze_result.append(
                [f"{frame_id:06d}", class_name, box[0], box[1], box[2], box[3], round(score.item(), 3), process_time])
        return analyze_result, process_time

    def analyze_single_segment(self, filename, frames_id):
        """analyze video segment and return the bboxes of each frames and processing time.
        @params:
            filename: video segment filename
            frames_id: List[int] index of each frame in the ground truth order
        @return:
            bboxes: BoundingBox of each detected frame
            processing_time: time to analyze the video segment
        """
        cap = cv2.VideoCapture(filename)
        frames = []
        bboxes = []
        processing_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        assert len(frames) == len(frames_id), "video capture wrong."
        for frame, frame_id in zip(frames, frames_id):
            result, process_time = self.analyze_single_frame(
                frame, frame_id, True)
            bboxes.append(self.prediction2bbox(result))
            processing_time += process_time
        return bboxes, processing_time

    def prediction2bbox(self, detection):
        bboxes = []
        for item in detection:
            bbox = BoundingBox.of_bbox(item[0], item[1], item[2],
                                       item[3], item[4], item[5], item[6])
            bboxes.append(bbox)
        return bboxes


if __name__ == '__main__':
    detr = DetrDetector()
    yolox = YoloDetector("yolov5n")
    gt_detr = Evaluator("acc", "detr", 1050)
    gt_yolo = Evaluator("acc", "yolov5x", 1050)
    result, _ = detr.detect("gt/1920x1080/000001.jpg", 1)
    result_yolo, _ = yolox.detect("gt/1920x1080/000001.jpg", 1)
    result = detr.prediction2bbox(result)
    result_yolo = yolox.prediction2bbox(result_yolo)
    result, mAp = gt_detr.evaluate(result, "1920x1080", 3)
    result_yolo, mAp_yolo = gt_yolo.evaluate(
        result_yolo, "1920x1080", 1)
    print(f"map: {mAp}/{mAp_yolo}")
