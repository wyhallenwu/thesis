from sim.network import NetworkSim
from sim.detectors import YoloDetector, DetrDetector
from sim.util import Evaluator
import random


class Server():
    def __init__(self, server_id: int, trace_path: str, model_type: str, gt_acc_path: str, frames_num: int) -> None:
        self.server_id = server_id
        self.network = NetworkSim(trace_path)
        self.model_type = model_type
        if self.model_type[:4] == "yolo":
            self.detector = YoloDetector(model_type)
        else:
            self.detector = DetrDetector()
        self.evaluator = Evaluator(gt_acc_path, "yolov5x", frames_num)
        self.rtt = random.randint(60, 80)
        self.process_chunks_ids = []

    def analyze_video_chunk(self, chunk_filename, frames_id, resolution):
        """current video chunk is processed by local device.
        @param:
            chunk_filename: path to the video chunk 
            frames_id: List[int] index of each frame
            resolution: [width, height]
        @return:
            results: wrapped result of earh frame(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps: mAp of each frame
            processing_time: process time of the whole chunk
        """
        self.process_chunks_ids.append(int(chunk_filename[:6]))
        bboxes, processing_time = self.detector.detect_video_chunk(
            chunk_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, f"{resolution[0]}x{resolution[1]}", f"{frame_id:06d}")
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def step_network(self):
        bws = self.network.step()
        throughputs = sum(bws)
        return bws, throughputs
