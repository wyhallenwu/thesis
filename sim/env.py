from typing import List
import numpy as np
from sim.util import energy_consuming
import torch
from sim.client import Client
import gymnasium
from gymnasium import spaces
from sim.server import Server, OffloadingTargets
from collections import OrderedDict
import time
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
# RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
# QUANTIZER = [5, 15, 25, 35, 45]
# SKIP = [0, 1, 2, 4, 5]
# FRAMERATE = [30, 15, 10, 6, 5]
RESOLUTION = [[1920, 1080], [1600, 900], [960, 540]]
QUANTIZER = [10, 25, 40]
SKIP = [0, 1, 2]
FRAMERATE = [30, 15, 10]
SERVER_NUM = 2
BYTES_IN_MB = 1000000
SKIP_THRESHOLD = 0.5
MAX_STEPS = 5000
LOG = "log/"
TRUNCATED_FRAMES_NUM = 10000
BW_NORM = 1e4
BWS_BOUND = 1e5


class SimEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, algorithm: str):
        # client
        self.client = Client(MOT_DATASET_PATH, GT_PATH,
                             TMP_PATH+"/"+algorithm, BUFFER_SIZE)
        # remote servers
        self.server1 = Server(1, "norway", "detr", GT_ACC_PATH, MOT_FRAMES_NUM)
        self.server2 = Server(2, "norway", "yolov5m",
                              GT_ACC_PATH, MOT_FRAMES_NUM)
        self.servers = OffloadingTargets([self.server1, self.server2])
        # other
        self.algorithm = algorithm
        self.steps_count = 0
        self.skipped_chunk_count = 0
        self.chunk_count = 0
        self.drain_mode = False
        self.log = LOG + algorithm + \
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".csv"
        self.remain_time = 0
        self.frames_sent = 0

        # envs
        self.actions_mapping = self.__build_actions_mapping(SERVER_NUM + 1)
        self.action_space = spaces.Discrete(len(self.actions_mapping))
        self.observation_space = spaces.Dict({
            "past_bws_mean": spaces.Box(low=np.array([0, 0]), high=np.array(self.servers.get_max_bw())/BYTES_IN_MB, dtype=np.float32),
            "past_bws_std": spaces.Box(low=np.array([0, 0]), high=np.array(self.servers.get_max_bw())/BYTES_IN_MB, dtype=np.float32),
            "available_buffer_size": spaces.Box(low=-2 * BUFFER_SIZE/BYTES_IN_MB, high=2 * BUFFER_SIZE / BYTES_IN_MB, dtype=np.float32),
            "chunk_size": spaces.Box(low=0, high=2.0 * BUFFER_SIZE / BYTES_IN_MB, dtype=np.float32),
            "past_chunk_delay": spaces.Box(low=0, high=np.inf, dtype=np.float32)
        })
        # self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -2.0 * BUFFER_SIZE, 0.0]),
        #                                     high=np.array([np.inf, np.inf, np.inf, np.inf, 2.0*BUFFER_SIZE, 2.0*BUFFER_SIZE], dtype=np.float64))

        print("BUILD SIMENV DONE.")

    def __build_actions_mapping(self, links) -> List:
        configs = []
        for resolution in RESOLUTION:
            for q in QUANTIZER:
                for framerate in FRAMERATE:
                    for id in range(links):
                        configs.append(
                            {"resolution": resolution, "framerate": framerate, "quantizer": q, "target": id})
        return configs

    def do_chunk_analyzing(self, config):
        """sending and analyzing a chunk.
        @params:
            config(Dict): resolution, framerate, quantizer, target
        @returns:
            results(List): List of metrics for each frame in the chunk
            mAps(List): mAp of each frame
            analyzing_time(float): detector's processing time(milliseconds)
            encoding_time(int): time for encoding the frames into a chunk(milliseconds)
            tranmission_time(float): transmission time for sending the chunk from client to server  
            [bws_mean1, bws_mean2, bws_std1, bws_std2]: mean and std of all offloading targets bandwidth to send the chunk
            chunk_index(int): the index of the chunk
            frames_id(List[int]): list of frames it contains
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
                bws = self.servers.step_networks()
                bws1.append(bws[0])
                bws2.append(bws[1])
                throughputs = bws[config["target"] - 1]
                t = 1 if throughputs <= bytes_to_send else bytes_to_send / throughputs
                transmission_time += server.rtt + int(t * 1000)
                bytes_to_send -= throughputs
        else:
            results, mAps, analyzing_time = self.client.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
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
            Dict:
                empty(bool): True if the client buffer is empty
                drain(bool):True if in draining_mode else False
                # results(List[float]): evaluated metrics of the chunk
                mAps(float): mean mAp of all frames in the chunk
                analyzing_time(int): millisecond that detectors takes to analyze the chunk
                encoding_time(int): millisecond that gstreamer encodes the chunk
                tranmission_time(int): millisecond to send the chunk to the remote target
                capture_chunk(int): num of chunks that should captured from the stream while sending the current chunk
                bws_mean1(float): mean bandwidth of remote target 1 to send the chunk
                bws_mean2(float): mean bandwidth of remote target 2 to send the chunk
                bws_std1(float): std of the bandwidth of remote target 1
                bws_std2(float): std of the bandwidth of remote target 2
        """
        if self.client.empty():
            self.remain_time = 0
            self.client.retrieve(config)
            self.chunk_count += 1
            bws = [[] for _ in range(len(self.servers))]
            for _ in range(2):
                curr_bws = self.servers.step_networks()
                bws[0].append(curr_bws[0])
                bws[1].append(curr_bws[1])
            # return {"empty": True, "drain": self.drain_mode, "mAps": 0,
            #         "bws_mean1": np.mean(bws[0]), "bws_mean2": np.mean(bws[1]),
            #         "bws_std1": np.std(bws[0]), "bws_std2": np.std(bws[1]), "chunk_size": 0,
            #         "processing_energy": 0, "tranmission_energy": 0}
            return {"empty": True, "drain": self.drain_mode, "mAps": 0,
                    "analyzing_time": 0, "encoding_time": 0,
                    "transmission_time": 0, "capture_chunks": 1,
                    "bws_mean1": np.mean(bws[0]), "bws_mean2": np.mean(bws[1]),
                    "bws_std1": np.std(bws[0]), "bws_std2": np.std(bws[1]), "chunk_index": -1,
                    "frames_num": 0, "chunk_size": 0,
                    "resolution": config["resolution"], "framerate": config["framerate"], "quantizer": config["quantizer"],
                    "target": config["target"], "remian_time": self.remain_time,
                    "available_buffer_size": self.client.get_buffer_vacancy(),
                    "processing_energy": 0, "transmission_energy": 0, "delay": 0}

        results, mAps, analyzing_time, encoding_time, tranmission_time, \
            [bws_mean1, bws_mean2, bws_std1, bws_std2], chunk_index, frames_id, \
            chunk_size, processing_energy, transmission_energy = self.do_chunk_analyzing(
                config)
        self.frames_sent += len(frames_id)
        self.remain_time += (tranmission_time / MILLS_PER_SECOND)
        capture_chunks = int(
            self.remain_time // 2) if not self.drain_mode else 0
        self.remain_time -= capture_chunks * 2
        for _ in range(capture_chunks):
            if not self.drain_mode and self.client.full():
                self.drain_mode = True
                self.remain_time = 0
            self.client.retrieve(config, self.drain_mode)
            self.chunk_count += 1
            self.skipped_chunk_count += 1 if self.drain_mode else 0
        if self.drain_mode and self.client.empty():
            self.drain_mode = False
        # has removed "results" attribute in the return state
        return {"empty": False, "drain": self.drain_mode, "mAps": np.mean(mAps),
                "analyzing_time": analyzing_time, "encoding_time": encoding_time,
                "transmission_time": tranmission_time, "capture_chunks": capture_chunks,
                "bws_mean1": bws_mean1, "bws_mean2": bws_mean2, "bws_std1": bws_std1,
                "bws_std2": bws_std2, "chunk_index": chunk_index,
                "frames_num": len(frames_id), "chunk_size": chunk_size,
                "resolution": config["resolution"], "framerate": config["framerate"], "quantizer": config["quantizer"],
                "target": config["target"], "remain_time": self.remain_time, "available_buffer_size": self.client.get_buffer_vacancy(),
                "processing_energy": processing_energy, "transmission_energy": transmission_energy,
                "delay": tranmission_time + encoding_time + analyzing_time}

    def _get_obs(self, state, config):
        """get observation."""
        obs = {"past_bws_mean": np.array([state["bws_mean1"]/BW_NORM, state["bws_mean2"]/BW_NORM]),
               "past_bws_std": np.array([state["bws_std1"]/BW_NORM, state["bws_std2"]/BW_NORM]),
               "available_buffer_size": np.array([self.client.get_buffer_vacancy() / BYTES_IN_MB]),
               "chunk_size": np.array([state["chunk_size"] / BYTES_IN_MB]),
               "past_chunk_delay": np.array([state["delay"] / MILLS_PER_SECOND])}
        # obs = np.array([state["bws_mean1"], state["bws_mean2"], state["bws_std1"], state["bws_std2"],
        #                 self.client.get_buffer_vacancy(), state["chunk_size"]], dtype=np.float64)
        return obs

    def _get_reward(self, state, config):
        """given the state and action, return the reward.
        """
        # tradeoff (accuracy, energy, delay) 2023-04-02
        # reward = (10 * state["mAps"]) ** 4 / frames_norm[config["framerate"]] -\
        #     (state["analyzing_time"] +
        #      state["transmission_time"] + state["encoding_time"]) / 25 - (state["processing_energy"] + state["transmission_energy"]) / 1e9
        # if state["drain"]:
        #     return -100
        # if state["empty"]:
        #     return -50
        # if state["target"] == 0:
        #     reward -= 100
        frames_norm = {10: 0.1, 15: 0.2, 30: 0.5}
        reward = (10 * state["mAps"]) ** 3 * \
            frames_norm[config["framerate"]] - \
            (state["delay"] / MILLS_PER_SECOND) - \
            (state["processing_energy"] + state["transmission_energy"]) / 1e9
        if state["drain"]:
            return -100
        elif state["empty"]:
            return -100
        elif state["target"] == 0:
            reward -= 50
        return reward

    def step(self, action):
        """given and perform an action and get information.
        @params:
            action(int): action generated by the algorithm
        @return:
            observation: past_average_bandwidth, client_buffer_vacancy, past_segment_size, past_framerate, past_quantizer, past_resolution
            reward(int): reward of current step
            terminated(bool): termination
            truncated(bool): truncation
            info(dict): state
        """
        self.steps_count += 1
        print(action)
        config = self.actions_mapping[action]
        state = self.update_state(config)
        obs = self._get_obs(state, config)
        reward = self._get_reward(state, config)
        truncated = self.truncated()
        terminated = self.terminated()
        with open(self.log, 'a') as f:
            for k, v in state.items():
                f.write(f"{k}: {v}, ")
            f.write('\n')
        return obs, reward, terminated, truncated, state

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.servers.reset()
        self.steps_count = 0
        self.skipped_chunk_count = 0
        self.chunk_count = 0
        self.drain_mode = False
        self.frames_sent = 0
        obs = OrderedDict()
        obs["past_bws_mean"] = np.array([0.0, 0.0], dtype=np.float32)
        obs["past_bws_std"] = np.array([0.0, 0.0], dtype=np.float32)
        obs["available_buffer_size"] = np.array(
            [self.client.get_buffer_vacancy() / BYTES_IN_MB], dtype=np.float32)
        obs["chunk_size"] = np.array([0], dtype=np.float32)
        obs["past_chunk_delay"] = np.array([0], dtype=np.float32)
        # obs = np.array(
        #     [0.0, 0.0, 0.0, 0.0, self.client.get_buffer_vacancy(), 0.0], dtype=np.float64)
        return obs, {}

    def truncated(self):
        return self.steps_count > MAX_STEPS or self.frames_sent > TRUNCATED_FRAMES_NUM

    def terminated(self):
        return self.client.done()

    def close(self):
        return
