import os
import numpy as np
from sim.detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator, energy_consuming
from sim.client import Client
import gymnasium
from gymnasium import spaces
from sim.server import Server, OffloadingTargets
from collections import OrderedDict
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
SKIP_THRESHOLD = 0.5
MAX_STEPS = 500
LOG = "log/"


class SimEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        # envs
        self.actions_mapping = self.__build_actions_mapping(SERVER_NUM + 1)
        self.action_space = spaces.Discrete(len(self.actions_mapping))
        self.observation_space = spaces.Dict({
            "past_bws_mean": spaces.Box(0, float('inf'), shape=(2, ), dtype=float),
            "past_bws_std": spaces.Box(0, float('inf'), shape=(2, ), dtype=float),
            "available_buffer_size": spaces.Box(-2 * BUFFER_SIZE, 2 * BUFFER_SIZE, dtype=int),
            "chunk_size": spaces.Box(0, 2 * BUFFER_SIZE, dtype=int)
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
        self.skipped_chunk_count = 0
        self.chunk_count = 0
        self.drain_mode = False
        self.log = LOG + \
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".json"
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
            bws1.append(self.servers.current_bws[0])
            bws2.append(self.servers.current_bws[1])
        return results, mAps, analyzing_time, encoding_time, transmission_time,\
            [np.mean(bws1), np.mean(bws2), np.std(bws1), np.std(bws2)], \
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
        # TODO:检查逻辑
        if self.client.empty() and self.drain_mode == False:
            self.client.retrieve(config)
            self.chunk_count += 1
            bws, throughputs = self.servers.step_networks()
            return {"empty": True, "drain": self.drain_mode, "mAps": 0,
                    "bws_mean1": np.mean(bws[0]), "bws_mean2": np.mean(bws[1]),
                    "bws_std1": np.std(bws[0]), "bws_std2": np.std(bws[1]), "chunk_size": 0}

        results, mAps, analyzing_time, encoding_time, tranmission_time, \
            [bws_mean1, bws_mean2, bws_std1, bws_std2], chunk_index, frames_id, \
            chunk_size, processing_energy, transmission_energy = self.do_chunk_analyzing(
                config)
        capture_chunks = int(
            tranmission_time / MILLS_PER_SECOND) // 2 if not self.drain_mode else 0
        for _ in range(capture_chunks):
            if self.client.full():
                self.drain_mode = True
            self.client.retrieve(config, self.drain_mode)
            self.chunk_count += 1
            self.skipped_chunk_count += 1 if self.drain_mode else 0
        if self.drain_mode and self.client.empty():
            self.drain_mode = False
        # remove "results" in the return state
        return {"empty": False, "drain": self.drain_mode, "mAps": mAps,
                "analyzing_time": analyzing_time, "encoding_time": encoding_time,
                "transmission_time": tranmission_time, "capture_chunks": capture_chunks,
                "bws_mean1": bws_mean1, "bws_mean2": bws_mean2, "bws_std1": bws_std1,
                "bws_std2": bws_std2, "chunk_index": chunk_index,
                "frames_id": frames_id, "chunk_size": chunk_size, "target": config["target"],
                "processing_energy": processing_energy, "tranmission_energy": transmission_energy}

    def _get_obs(self, state, config):
        obs = {"past_bws_mean": np.array([state["bws_mean1"], state["bws_mean2"]]),
               "past_bws_std": np.array([state["bws_std1"], state["bws_std2"]]),
               "available_buffer_size": np.array([self.client.get_buffer_vacancy()]),
               "chunk_size": np.array(state["chunk_size"])}
        return obs

    def _get_reward(self, state, config):
        # TODO: improve reward design
        resolution_reward = {1920: 4, 1600: 2, 1280: 1, 960: -2}
        quantizer_reward = {5: -5, 15: 1, 25: 3, 35: 4, 45: 5}
        framerate_reward = {30: 5, 15: 3, 10: 1, 6: -3, 5: -5}
        reward = (resolution_reward[config["resolution"][0]] + quantizer_reward[config["quantizer"]] +
                  framerate_reward[config["framerate"]]) * np.mean(state["mAps"])
        if state["drain"]:
            return -10
        if state["empty"]:
            return -2
        if state["target"] == 0:
            return -abs(reward)
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
            f.write(json.dumps(state, indent=2))
        return obs, reward, done, truncated, state

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.servers.reset()
        self.steps_count = 0
        self.skipped_chunk_count = 0
        self.chunk_count = 0
        self.drain_mode = False
        obs = OrderedDict()
        obs["past_bws_mean"] = np.array([0, 0])
        obs["past_bws_std"] = np.array([0, 0])
        obs["available_buffer_size"] = np.array(
            [self.client.get_buffer_vacancy()])
        obs["chunk_size"] = np.array([0])
        return obs, {}

    def truncated(self):
        return self.steps_count > MAX_STEPS or (self.steps_count > MAX_STEPS / 2 and
                                                self.skipped_chunk_count / self.steps_count > SKIP_THRESHOLD)

    def done(self):
        return self.client.done()

    def close(self):
        return
