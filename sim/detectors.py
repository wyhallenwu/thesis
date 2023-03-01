import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time


class YoloDetector():
    def __init__(self, model_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.model = torch.hub.load(
            'ultralytics/yolov5', self.model_type, pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.frame_counter = 0

    def detect(self, frame, frame_id=None):
        """detect single frame and return the result.
        @params:
            frame: filename or cv2 image or pytorch image or numpy image or PIL image
        @return:
            result: processing_time, confidence, boxes
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
        detection_result = [[frame_id, class_name[i], round(x_min[i], 3), round(y_min[i], 3),
                             round(x_max[i], 3), round(y_max[i], 3), round(
                                 confidence[i], 3), round(process_time, 3)] for i in range(len(confidence))]
        return detection_result

    def reset(self):
        self.frame_counter = 0

    # def save_single_frame(self, video_path):
    #     _, _, files = os.walk(video_path)
    #     crf_config = []
    #     for video_file in tqdm(files, desc="processing"):
    #         config = video_file[:-4].split('_')
    #         width = int(config[0])
    #         frame_rate = int(config[1])
    #         constant_rate_factor = int(config[2])
    #         frame_counter = 0
    #         config = [width, frame_rate, constant_rate_factor]
    #         if config not in crf_config:
    #             crf_config.append(config)
    #             print(
    #                 f"split and saving file in config: {config[0]}:{config[1]}:{config[2]}")
    #             path = video_path+'/'+f"{config[0]}_{config[1]}_{config[2]}"
    #             if not os.path.exists(path):
    #                 print(f"making new folder {path}")
    #                 os.makedirs(path)
    #             cap = cv2.VideoCapture(video_path + "/" + video_file)
    #             while(cap.isOpened()):
    #                 _, frame = cap.read()
    #                 if frame is not None:
    #                     frame_counter += 1
    #                     cv2.imwrite(
    #                         f"{path}/{frame_counter:06d}.jpg", frame)
    #                 else:
    #                     break
    #             cap.release()


class DetrDetector():
    def __init__(self, threshold=0.75) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101").to(self.device)
        self.threshold = threshold
        self.model_type = "detr"

    def detect(self, frame, frame_id):
        frame = Image.open(frame)
        inputs = self.processor(
            images=frame, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([frame.size[::-1]]).to(self.device)
        start_time = time.time()
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold)[0]
        end_time = time.time()
        process_time = round((end_time - start_time), 3)
        detection_result = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 3) for i in box.tolist()]
            detection_result.append(
                [frame_id, label.item(), box[0], box[1], box[2], box[3], round(score.item(), 3), process_time])
        return detection_result

# if __name__ == "__main__":
#     # run: python pre_detect.py --filepath=test_data/test.flv
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_type", type=str,
#                         help="yolov5n, yolov5s, yolov5m, yolov5l, yolov5x")
#     parser.add_argument("--filepath", type=str, help="test video file path")
#     parser.add_argument("--video_path", type=str,
#                         help="videos path for save single images")
#     parser.add_argument("--saving_path", type=str,
#                         help="pre-detection csv saving path")
#     args = parser.parse_args()
#     # if args.model_type:
#     #     if args.filepath:
#     #         detector = Detector(args.model_type)
#     #         detector.test(args.filepath)
#     #     if args.video_path and args.saving_path is None:
#     #         detector = Detector(args.model_type)
#     #         detector.save_single_frame(args.video_path)
#     #     if args.saving_path and args.video_path:
#     #         detector = Detector(args.model_type)
#     #         detector.pre_detect(args.video_path, args.saving_path)
