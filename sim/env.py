import numpy as np
import os
import time
# action: [framerate, resolution, quantizer, offloading target]


class SimEnv():
    def __init__(self, config):
        self.config = config
        self.action_space = None
        self.buffer_size = self.config["buffer_size"]
        self.current_frame_index = 0
        self.detector = None
        self.replay_buffer = []
        self.inter_path = self.config["inter_path"]
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
