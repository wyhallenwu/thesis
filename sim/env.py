import numpy as np
import os
import time
from detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator
from client import Client
import gym
from gym import spaces
from network import Networks
# action: [framerate, resolution, quantizer, offloading target]
# skip = [0, 1, 2, 4, 5] => fps = [30, 15, 10, 6, 5]
BUFFER_SIZE = 10  # n seconds video chunks
DATASET_PATH = "MOT16-04"
TMP_PATH = "tmp"
TMP_FRAMES = TMP_PATH + "/frames"
TMP_CHUNKS = TMP_PATH + "/chunks"
GT_ACC_PATH = "acc"
FCC_PATH = "202201cooked"
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
QUANTIZER = [5, 15, 25, 35, 45]
SKIP = [0, 1, 2, 4, 5]


class SimEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.action_mapping = []
        for resolution in RESOLUTION:
            for q in QUANTIZER:
                for skip in SKIP:
                    self.action_mapping.append([resolution, q, skip])
        # [skip(5), resolution(4), quantizer(5), target(n)] 5*4*5*n
        self.action_space = spaces.Discrete(len(self.action_mapping))
        self.detector = DetrDetector()
        self.gt = Evaluator(GT_ACC_PATH, self.detector.model_type, 1050)
        self.tmp_path = TMP_PATH
        self.client = Client(DATASET_PATH, TMP_PATH)
        self.policy = {"skip": 0, "resolution": [
            1920, 1080], "quantizer": 20, "target": 0}
        # network
        self.servers_num = self.config["servers_num"]
        self.networks = Networks(self.servers_num, FCC_PATH)

        # congestion indication
        self.drain_buffer = False

    def step(self, action):
        self.clean_tmp_frames()
        if self.drain_buffer:
            self.drain()
        self.take_action(action)

    def get_award(self, event):
        if event == "buffer_full":
            return -100

    def take_action(self, action):
        if action is None:
            action = self.policy
        # capture the frames
        # if buffer is full, wait until the buffer is drained
        current_bws = self.networks.next_bws()
        buffer_full = self.client.retrieve(action["skip"])

    def drain(self):
        if self.client.empty():
            self.drain_buffer = False
        pass

    def reset(self):
        pass

    def clean_tmp_frames(self):
        os.system(f"rm -rf {self.client.tmp_frames}/*")

    def generate_observation(self):
        pass
