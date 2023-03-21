import torch
import cv2
from podm.metrics import BoundingBox
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
from util import Evaluator


class YoloDetector():
    """YoloDetector"""

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

    def detect(self, frame, frame_id):
        """detect single frame and return the result.
        @params:
            frame: filename
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
        detection_result = [[frame_id, class_name[i].replace(' ', '_'), round(x_min[i], 3), round(y_min[i], 3),
                             round(x_max[i], 3), round(y_max[i], 3), round(
                                 confidence[i], 3), round(process_time, 3)] for i in range(len(confidence))]
        return detection_result, round(process_time, 3)

    def detect_video_chunk(self, filename, frames_id):
        """analyze video chunk and return a List of bboxes of each frame and the processing time of the chunk."""
        cap = cv2.VideoCapture(filename)
        frames = []
        results = []
        processing_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            assert ret, "read video failed."
            frames.append(frame)
        cap.release()
        for frame, frame_id in zip(frames, frames_id):
            result, process_time = self.detect(frame, frame_id)
            results.append(self.prediction2bbox(result))
            processing_time += process_time
        return results, processing_time

    def reset(self):
        self.frame_counter = 0

    def prediction2bbox(self, detection):
        bboxes = []
        for item in detection:
            bbox = BoundingBox.of_bbox(item[0], item[1], item[2],
                                       item[3], item[4], item[5], item[6])
            bboxes.append(bbox)
        return bboxes


class DetrDetector():
    def __init__(self, threshold=0.8) -> None:
        """
        @params:
            threshold: filter the result which confidence is lower the threshold
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101").to(self.device)
        self.threshold = threshold
        self.model_type = "detr"
        print("BUILD DETR DETECTOR DONE.")

    def detect(self, frame, frame_id, video=False):
        """detect a frame.
        @params:
            frame(str): the filename of the image
            frame_id(str): the index of the frame starting from 1 in the format :06d
        """
        if not video:
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
        detection_result = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 3) for i in box.tolist()]
            class_name = str(
                self.model.config.id2label[label.item()]).replace(' ', '_')
            detection_result.append(
                [frame_id, class_name, box[0], box[1], box[2], box[3], round(score.item(), 3), process_time])
        return detection_result, process_time

    def detect_video_chunk(self, filename, frames_id):
        cap = cv2.VideoCapture(filename)
        frames = []
        results = []
        processing_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frames.append(frame)
        cap.release()
        for frame, frame_id in zip(frames, frames_id):
            result, process_time = self.detect(frame, frame_id, True)
            results.append(self.prediction2bbox(result))
            processing_time += process_time
        return results, processing_time

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
    gt = Evaluator("acc", "detr", 1050)
    gt_yolo = Evaluator("acc", "yolov5x", 1050)
    result, _ = detr.detect("gt/1920x1080/000001.jpg", "000001")
    result_yolo, _ = yolox.detect("gt/1920x1080/000001.jpg", "000001")
    result = detr.prediction2bbox(result)
    result_yolo = yolox.prediction2bbox(result_yolo)
    result, mAp = gt.evaluate(result, "1920x1080", "000001")
    result_yolo, mAp_yolo = gt_yolo.evaluate(
        result_yolo, "1920x1080", "000001")
    print(f"map: {mAp}/{mAp_yolo}")
    for cls, metric in result:
        label = metric.label
        print('ap', metric.ap)
        print('precision', metric.precision)
        print('interpolated_recall', metric.interpolated_recall)
        print('interpolated_precision', metric.interpolated_precision)
        print('tp', metric.tp)
        print('fp', metric.fp)
        print('num_groundtruth', metric.num_groundtruth)
        print('num_detection', metric.num_detection)

    for cls, metric in result_yolo:
        label = metric.label
        print('ap', metric.ap)
        print('precision', metric.precision)
        print('interpolated_recall', metric.interpolated_recall)
        print('interpolated_precision', metric.interpolated_precision)
        print('tp', metric.tp)
        print('fp', metric.fp)
        print('num_groundtruth', metric.num_groundtruth)
        print('num_detection', metric.num_detection)

    results, processing_time = yolox.detect_video_chunk(
        "10.flv", [f"{i:06d}" for i in range(1, 11)])
    for result, id in zip(results, [f"{i:06d}" for i in range(1, 11)]):
        r, mAp = gt_yolo.evaluate(result, id)
        print(mAp)
