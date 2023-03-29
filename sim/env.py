import os
import numpy as np
from sim.detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator, energy_consuming
from sim.client import Client
import gymnasium
from gymnasium import spaces
from sim.server import Server, OffloadingTargets
import itertools
import time
import json
# action: [framerate, resolution, quantizer, offloading target]
# skip = [0, 1, 2, 4, 5] => fps = [30, 15, 10, 6, 5]
# timestamp is 2 seconds
MILLS_PER_SECOND = 1000
BUFFER_SIZE = 2000000  # bytes = 2Mb
MOT_DATASET_PATH = "MOT16-04/img1"
GT_PATH = "gt"
MOT_FRAMES_NUM = 1050
TMP_PATH = "tmp"
TMP_FRAMES = TMP_PATH + "/frames"
TMP_CHUNKS = TMP_PATH + "/chunks"
GT_ACC_PATH = "acc"
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
QUANTIZER = [5, 15, 25, 35, 45]
SKIP = [0, 1, 2, 4, 5]
FRAMERATE = [30, 15, 10, 6, 5]
SERVER_NUM = 2
BYTES_IN_MB = 1000000
SKIP_THRESHOLD = 10
MAX_STEPS = 500
LOG = "log/"


class SimEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        # envs
        self.actions_mapping = self.__build_actions_mapping(SERVER_NUM + 1)
        self.action_space = spaces.Discrete(len(self.actions_mapping))
        # self.observation_space = spaces.MultiDiscrete(
        #     [BANDWIDTH_BOUND, CLIENT_BUFFER_SIZE, CLIENT_BUFFER_SIZE, 5, 5, 4, 1])
        self.observation_space = spaces.Dict({
            "past_bws_mean": spaces.Box(0, float('inf'), shape=(2,), dtype=int),
            "past_bws_std": spaces.Box(0, float('inf'), shape=(2, ), dtype=float),
            "avaliable_buffer_size": spaces.Box(float('inf'), float('inf'), dtype=int),
            "chunk_size": spaces.Box(0, float('inf'), dtype=int)
        })
        # client
        self.client = Client(MOT_DATASET_PATH, GT_PATH, TMP_PATH, BUFFER_SIZE)
        # remote servers
        self.server1 = Server(1, "norway", "detr", GT_ACC_PATH, MOT_FRAMES_NUM)
        self.server2 = Server(2, "norway", "yolov5m",
                              GT_ACC_PATH, MOT_FRAMES_NUM)
        self.servers = OffloadingTargets([self.server1, self.server2])
        # other
        self.steps_count = 0
        self.skipped_capture_count = 0
        self.drain_status = False
        self.log = LOG + \
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime() + ".json")
        print("BUILD SIMENV DONE.")

    def __build_actions_mapping(self, links):
        configs = []
        for resolution in RESOLUTION:
            for q in QUANTIZER:
                for framerate in FRAMERATE:
                    for id in range(links):
                        configs.append(
                            {"resolution": resolution, "framerate": framerate, "quantizer": q, "target": id})
        return configs

    def do_chunk_analyzing(self, config):
        """sending and analyzing a single chunk.
        @params:
            config:Dict[resolution, framerate, quantizer, target]
        @returns:
            results(List): metrics evaluation result of each frame in the chunk
            mAps(List): mAp of each frame
            analyzing_time(float): analyzing time of the chunk
            encoding_time(int): milliseconds time for encoding the frames into a chunk
            tranmission_time(float): transmission time for sending the chunk from client to server  
            []: mean and std of all servers' bws to send the chunk
            chunk_index(int): the index of the chunk
            frames_id(List): list of frames it contains
            chunk_size(int): bytes of the chunk
            local_processing_energy(float): if chunk is processed locally
            tranmission_energy(float): if chunk is sent to remote server
        """
        chunk_index, frames_id, chunk_size, encoding_time, resolution = self.client.get_chunk()
        transmission_time = 0
        local_processing_energy, transmission_energy = energy_consuming(
            len(frames_id), config["resolution"], config["target"] == 0)
        bws1 = []
        bws2 = []
        if config["target"] != 0:
            server = self.servers.get_server_by_id(config["target"])
            bytes_to_send = chunk_size
            results, mAps, analyzing_time, = server.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
            while bytes_to_send > 0:
                bw, throughputs = self.servers.step_networks()
                bws1.extend(bw[0])
                bws2.extend(bw[1])
                throughputs = throughputs[config["target"] - 1]
                t = 1 if throughputs <= bytes_to_send else bytes_to_send / throughputs
                transmission_time += server.rtt + int(t * 1000)
                bytes_to_send -= throughputs
        else:
            results, mAps, analyzing_time = self.client.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
            # TODO: check
            bws1.extend(self.servers.current_bws[0])
            bws2.extend(self.servers.current_bws[1])
        return results, mAps, analyzing_time, encoding_time, transmission_time,\
            [int(np.mean(bws1), int(np.mean(bws2))), round(np.std(bws1), 3), round(np.std(bws2), 3)], \
            chunk_index, frames_id, chunk_size, local_processing_energy, transmission_energy

    def update_state(self, config):
        """given an config(action) and update the environment.
        @params:
            config: Dict[resolution, framerate, quantizer, target]
        @returns:
            Dict of the following attributes 
            "drain"(bool):if True in draining buffer else False
            "results"(List): metrics of the chunk
            "mAps"(float): mAps of the chunk
            "analyzing_time"(int): ms that detectors takes to analyze the chunk
            "encoding_time"(int): ms that gstreamer encodes the chunk
            "tranmission_time"(int): ms to send the chunk to the remote
            "capture_chunk"(int): num of chunks that would retrieve from the stream
            "average_bws"(int): average bandwidth to send the chunk  
        """
        # capture from video stream
        if self.drain_status == False:
            capture_success = self.client.retrieve(config)
            if not capture_success:
                self.drain_status = True
        results, mAps, analyzing_time, encoding_time, tranmission_time, \
            [bws_mean1, bws_mean2, bws_std1, bws_std2], chunk_index, frames_id, \
            chunk_size, processing_energy, transmission_energy = self.do_chunk_analyzing(
                config)
        capture_chunks = int(
            tranmission_time / MILLS_PER_SECOND) // 2 if not self.drain_status else 0
        # TODO:如果是drain buffer状态，跳过这几个chunk
        # 添加处理empty buffer的情况
        if self.drain_status:
            self.client.capture()
            for _ in range(capture_chunks):
                self.client.retrieve(config["framerate"])
        if self.drain_status and self.client.empty():
            self.drain_status = False
        return {"drain": self.drain_status, "results": results, "mAps": mAps,
                "analyzing_time": analyzing_time, "encoding_time": encoding_time,
                "transmission_time": tranmission_time, "capture_chunks": capture_chunks,
                "bws_mean1": bws_mean1, "bws_mean2": bws_mean2, "bws_std1": bws_std1,
                "bws_std2": bws_std2, "chunk_index": chunk_index,
                "frames_id": frames_id, "chunk_size": chunk_size, "target": config["target"],
                "processing_energy": processing_energy, "tranmission_energy": transmission_energy}

    def _get_obs(self, state, config):
        """
            "past_bws_mean": spaces.Box(0, float('inf'), shape=(2,), dtype=int),
            "past_bws_std": spaces.Box(0, float('inf'), shape=(2, ), dtype=float),
            "avaliable_buffer_size": spaces.Box(float('inf'), float('inf'), dtype=int),
            "chunk_size": spaces.Box(0, float('inf'), dtype=int)
        """
        obs = {"past_bws_mean": np.array([state["bw_mean1"], state["bws_mean2"]]),
               "past_bws_std": np.array([state["bws_std1"], state["bws_std2"]]),
               "avaliable_buffer_size": np.array([self.client.get_buffer_vacancy()]),
               "chunk_size": np.array(state["chunk_size"])}
        return obs

    def _get_reward(self, state, config):
        # TODO: improve reward design
        resolution_reward = {1920: 5, 1600: 4, 1280: 3, 960: 2}
        quantizer_reward = {5: 1, 15: 2, 25: 3, 35: 4, 45: 5}
        framerate_reward = {30: 5, 15: 3, 10: 2, 6: 1, 5: 0}
        reward = (resolution_reward[config["resolution"][0]] + quantizer_reward[config["quantizer"]] +
                  framerate_reward[config["framerate"]]) * np.mean(state["mAps"])
        if state["drain"]:
            return -10
        if state["target"] == 0:
            return -5
        print("reward: ", reward)
        return reward

    def step(self, action):
        """given and perform an action and get information.
        @params:
            action(int): action generated by the algorithm
        @return:
            observation: past_average_bandwidth, client_buffer_vacancy, past_segment_size, past_framerate, past_quantizer, past_resolution
            reward(int):
            terminated(bool):
            truncated(bool):
            info(dict):
        """
        self.steps_count += 1
        config = self.actions_mapping[action]
        state = self.update_state(config)
        obs = self._get_obs(state, config)
        reward = self._get_reward(state, config)
        truncated = self.truncated()
        done = self.done()
        with open(self.log, 'a') as f:
            f.write(json.dumps(state, indent=4))
        return obs, reward, done, truncated, state

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.server1.reset()
        self.server2.reset()
        self.steps_count = 0
        self.skipped_capture_count = 0
        self.drain_status = False
        return self.observation_space.sample(), {}

    def truncated(self):
        return self.steps_count > MAX_STEPS or self.skipped_capture_count > SKIP_THRESHOLD

    def done(self):
        return self.client.done()

    def close(self):
        return
