import numpy as np
import os
import time
from detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator
# action: [framerate, resolution, quantizer, offloading target]
# skip = [0, 1, 2, 4, 5] => fps = [30, 15, 10, 6, 5]
BUFFER_SIZE = 10  # n seconds video chunks
TMP_PATH = "tmp/"
GT_ACC_PATH = "acc"


class SimEnv():
    def __init__(self, config):
        self.config = config
        self.action_space = None
        self.buffer_size = BUFFER_SIZE
        self.current_frame_index = 0
        self.detector = DetrDetector()
        self.gt = Evaluator(GT_ACC_PATH, self.detector.model_type, 1050)
        self.tmp_path = TMP_PATH
        self.policy = {"framerate": 30, "resolution": [
            1920, 1080], "quantizer": 20}

    def step(self, action):
        self.clean_inter_path()

    def take_action(self, action):
        pass

    def reset(self):
        pass

    def init(self):
        pass

    def close(self):
        pass

    def clean_inter_path(self):
        os.system(f"rm -rf {self.inter_path}")
        os.makedirs(self.inter_path)

    def generate_observation(self):
        pass
