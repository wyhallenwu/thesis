from typing import List
from sim.network import NetworkSim
from sim.detectors import YoloDetector, DetrDetector
from sim.util import Evaluator
import random


class Server():
    """Server is the simulated edge server."""

    def __init__(self, server_id: int, traces: str, model_type: str, gt_acc_path: str, frames_num: int) -> None:
        self.server_id = server_id
        self.network = NetworkSim(traces)
        self.model_type = model_type
        if self.model_type[:4] == "yolo":
            self.detector = YoloDetector(model_type)
        else:
            self.detector = DetrDetector()
        self.evaluator = Evaluator(gt_acc_path, "yolov5x", frames_num)
        # simulated rtt delay
        self.rtt = random.randint(60, 80)

    def reset(self):
        self.rtt = random.randint(60, 80)
        self.network.reset()

    def analyze_video_segment(self, segment_filename, frames_id, resolution):
        """analyzing video segment and compare it with corresponding resolution preprocessed groundtruth.
        @param:
            segment_filename(str): path to the video segment 
            frames_id(List[int]): index of each frame
            resolution(List[int]): [width, height]
        @return:
            results(List): wrapped result of earh frame(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps(List[float]): mAp of each frame
            processing_time(float): process time of the whole segment
        """
        bboxes, processing_time = self.detector.analyze_single_segment(
            segment_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, resolution, frame_id)
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def step_network(self):
        """step the simulated network trace to next the timestamp.
        @returns:
            bws(List[int]): bytes per second
            throughputs(int): sum of bytes in the period
        """
        bw = self.network.step()
        return bw

    def get_max_bw(self):
        return self.network.get_max_bw()


class OffloadingTargets():
    """OffloadingTarges is the collection of edge servers."""

    def __init__(self, servers: List[Server]) -> None:
        self.servers = servers
        self.current_bws = [0] * len(self.servers)

    def add(self, server: Server):
        self.servers.append(server)

    def step_networks(self):
        """step the network of all offloading targets to the next the timestamp.
        @retuns:
            bws(List[List[int]]): bytes per second of each target in the eplased time.
            throughputs(List[int]): throughputs of each target in the eplased time.
        """
        bws = []
        for id, server in enumerate(self.servers):
            bw = server.step_network()
            bws.append(bw)
            self.current_bws[id] = bw
        return bws

    def __len__(self):
        return len(self.servers)

    def get_server_by_id(self, id):
        return self.servers[id - 1]

    def get_current_bw_by_id(self, id):
        return self.current_bws[id - 1]

    def get_max_bw(self):
        return [server.get_max_bw() for server in self.servers]

    def reset(self):
        for server in self.servers:
            server.reset()
        self.current_bws = [0] * len(self.servers)
