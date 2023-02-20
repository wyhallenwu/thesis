import numpy as np


class SimEnv():
    def __init__(self, config):
        self.config = config
        self.action_space = None
        self.buffer_size = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def init(self):
        pass

    def close(self):
        pass

    def generate_observation(self):
        pass