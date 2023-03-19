import os
from sim.detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator
from sim.client import Client
import gym
from gym import spaces
from sim.network import Networks
# action: [framerate, resolution, quantizer, offloading target]
# skip = [0, 1, 2, 4, 5] => fps = [30, 15, 10, 6, 5]
# timestamp is 2 seconds
BUFFER_SIZE = 10  # n seconds video chunks
DATASET_PATH = "MOT16-04"
TMP_PATH = "tmp"
TMP_FRAMES = TMP_PATH + "/frames"
TMP_CHUNKS = TMP_PATH + "/chunks"
GT_ACC_PATH = "acc"
FCC_PATH = "dataset/fcc/202201cooked"
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
QUANTIZER = [5, 15, 25, 35, 45]
SKIP = [0, 1, 2, 4, 5]
FRAMERATE = [30, 15, 10, 6, 5]
CLIENT_BUFFER_SIZE = 1000
SERVER_NUM = 2


class SimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        # networks
        self.servers_num = SERVER_NUM
        self.networks = Networks(self.servers_num, FCC_PATH)
        self.actions_config = self.form_actions(self.servers_num)
        # [skip(5), resolution(4), quantizer(5), links(n)] 5*4*5*n + local
        self.action_space = spaces.Discrete(len(self.actions_config) + 1)
        # TODO: observation spaces
        self.observation_space = spaces.Dict(
            {"past_throughput": spaces.Box(low=0, high=100000, dtype=int),
             "client_buffer_size": spaces.Box(low=0, high=CLIENT_BUFFER_SIZE, dtype=int),
             "past_segment_size": spaces.Box(low=0, high=100000),
             "past_framerate": spaces.Discrete(5),
             "past_quantizer": spaces.Discrete(5),
             "past_resolution": spaces.Discrete(5)})
        # detection and evaluator
        self.detector = DetrDetector()
        self.gt = Evaluator(GT_ACC_PATH, self.detector.model_type, 1050)
        self.tmp_path = TMP_PATH
        self.client = Client(DATASET_PATH, TMP_PATH)
        # default policy
        self.policy = {"skip": 0, "resolution": [
            1920, 1080], "quantizer": 20, "target": 0}
        # transmission
        self.current_chunk_index = -1
        # congestion indication
        self.drain_buffer = False
        print("BUILD SIMENV DONE.")

    def form_actions(self, links):
        configs = []
        for resolution in RESOLUTION:
            for q in QUANTIZER:
                for framerate in FRAMERATE:
                    for id in range(links):
                        configs.append([resolution, framerate, q, id])
        return configs

    def step(self, action):
        print("test")
        # self.clean_tmp_frames()
        # # if buffer is full, wait until the buffer is drained
        # if self.drain_buffer:
        #     self.drain(action)
        # else:
        #     self.take_action(action)
        return {"info": 100}, 1, True, {}

    def _get_obs(self):
        pass

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

    def drain(self, action):
        if self.client.empty():
            self.drain_buffer = False
        frames = self.client.get_frames_buffer()
        chunk_index, chunk_size, gst_time = self.client.process_video(
            frames, [action["resolution"], action["quantizer"]])
        self.current_chunk_index = chunk_index
        # download current chunk
        # TODO: 每一个timestamp的选择，ms/s
        sent = 0
        delay = 0
        while sent < chunk_size:
            sent += self.networks.next_bws()[action["target"]]

    def reset(self, seed=None):
        super().reset(seed=seed)
        pass

    def clean_tmp_frames(self):
        os.system(f"rm -rf {self.client.tmp_frames}/*")

    def generate_observation(self):
        pass
