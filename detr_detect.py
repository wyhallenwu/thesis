from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
from tqdm import tqdm
import cv2
import argparse
import time


class DetrDetector():
    def __init__(self, threshold=0.9) -> None:
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101")
        self.threshold = threshold

    def detect(self, video_path, saving_path):
        _, dirs, _ = next(os.walk(video_path))
        for dir in tqdm(dirs, desc="processing"):
            frame_counter = 0
            if not os.path.exists(f"{saving_path}/{dir}"):
                os.makedirs(f"{saving_path}/{dir}")
                print(f"makeing new folder: {saving_path}/{dir}")
            with open(f"{saving_path}/{dir}/{dir}_detr.csv", 'w') as f:
                for frame_name in sorted(os.listdir(f"{video_path}/{dir}")):
                    frame_counter += 1
                    image = Image.open(f"{video_path}/{dir}/{frame_name}")
                    inputs = self.processor(
                        images=image, return_tensors="pt")
                    outputs = self.model(**inputs)
                    # convert outputs (bounding boxes and class logits) to COCO API
                    # let's only keep detections with score > 0.9
                    target_sizes = torch.tensor([image.size[::-1]])
                    start_time = time.time()
                    results = self.processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=self.threshold)[0]
                    end_time = time.time()
                    process_time = end_time - start_time
                    # write result
                    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                        box = [round(i, 2) for i in box.tolist()]
                        # print(
                        #     f"Detected {self.model.config.id2label[label.item()]} with confidence "
                        #     f"{round(score.item(), 3)} at location {box}"
                        # )
                        f.write(
                            f"{frame_counter:06d} {process_time} {round(score.item(), 3)} {self.model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="video path for detection")
    parser.add_argument("--saving_path", type=str, help="result saving path")
    args = parser.parse_args()
    if args.video_path and args.saving_path:
        detr = DetrDetector()
        detr.detect(args.video_path, args.saving_path)
    