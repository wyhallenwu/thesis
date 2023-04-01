from typing import List
from sim.network import NetworkSim
from sim.detectors import YoloDetector, DetrDetector
from sim.util import Evaluator
import random


class Server():
    def __init__(self, server_id: int, traces: str, model_type: str, gt_acc_path: str, frames_num: int) -> None:
        self.server_id = server_id
        self.network = NetworkSim(traces)
        self.model_type = model_type
        if self.model_type[:4] == "yolo":
            self.detector = YoloDetector(model_type)
        else:
            self.detector = DetrDetector()
        self.evaluator = Evaluator(gt_acc_path, "yolov5x", frames_num)
        self.rtt = random.randint(60, 80)

    def reset(self):
        self.rtt = random.randint(60, 80)
        self.network.reset()

    def analyze_video_chunk(self, chunk_filename, frames_id, resolution):
        """analyzing video chunk and compare it with corresponding resolution preprocessed groundtruth.
        @param:
            chunk_filename(str): path to the video chunk 
            frames_id(List[int]): index of each frame
            resolution(List[int]): [width, height]
        @return:
            results(List): wrapped result of earh frame(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps(List[float]): mAp of each frame
            processing_time(float): process time of the whole chunk
        """
        bboxes, processing_time = self.detector.detect_video_chunk(
            chunk_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, f"{resolution[0]}x{resolution[1]}", frame_id)
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def step_network(self):
        """step the simulated network trace to next timestamp.

        @returns:
            bws(List[int]): bytes per second
            throughputs(int): sum of bytes in the period
        """
        bws = self.network.step()
        throughputs = sum(bws)
        return bws, throughputs


class OffloadingTargets():
    def __init__(self, servers: List[Server]) -> None:
        self.servers = servers
        self.current_bws = [0] * len(self.servers)

    def add(self, server: Server):
        self.servers.append(server)

    def step_networks(self):
        """step all offloading targets to the next timestamp.
        @retuns:
            bws(List[List[int]]): bytes per second of each target in the eplased time.
            throughputs(List[int]): throughputs of each target in the eplased time.
        """
        bws, throughputs = [], []
        for id, server in enumerate(self.servers):
            bw, throughput = server.step_network()
            bws.append(bw)
            self.current_bws[id] = bw
            throughputs.append(throughput)
        return bws, throughputs

    def get_server_by_id(self, id):
        return self.servers[id - 1]

    def get_current_bw_by_id(self, id):
        return self.current_bws[id - 1]

    def reset(self):
        for server in self.servers:
            server.reset()
        self.current_bws = [0] * len(self.servers)
