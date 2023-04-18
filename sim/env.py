from typing import List
import numpy as np
from sim.util import energy_consuming
import pandas as pd
from sim.client import Client
import gymnasium
from gymnasium import spaces
from sim.server import Server, OffloadingTargets
from collections import OrderedDict
import time
""" settings:
action space: [framerate, resolution, quantizer, offloading target]
skip [0, 1, 2, 4, 5] frames => fps [30, 15, 10, 6, 5]
default timestamp interval is 2 seconds(assume the stream is captured in 30fps)

setting option:
# RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
# QUANTIZER = [18, 23, 28, 33, 38, 43]
# SKIP = [0, 1, 2, 4, 5]
# FRAMERATE = [30, 15, 10, 6, 5]
"""
# ==================================hyperparameters===============================
MILLS_PER_SECOND = 1000
BUFFER_SIZE = 5000000  # bytes => 5Mb
MOT_FRAMES_NUM = 1050  # MOT16-04 has 1050 frames
TRUNCATED_FRAMES_NUM = 10000  # captured more than 10000 frames, truncated the episode
BW_NORM = 1e4  # scaling the bandwidth in observation
SERVER_NUM = 2  # num of remote servers
BYTES_IN_MB = 1000000
MAX_STEPS = 1000  # max steps in an episode

# paths
MOT_DATASET_PATH = "MOT16-04/img1"
GT_PATH = "gt"
TMP_PATH = "tmp"
GT_ACC_PATH = "acc"
LOG = "log/"

# action space
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
QUANTIZER = [18, 23, 28, 33, 38, 43]
SKIP = [0, 1, 2, 4, 5]
FRAMERATE = [30, 15, 10, 6, 5]
# ================================================================================


class SimEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, algorithm: str):
        # client
        self.client = Client(MOT_DATASET_PATH, GT_PATH,
                             TMP_PATH+"/"+algorithm, BUFFER_SIZE)
        # remote edge servers
        self.server1 = Server(1, "norway", "detr", GT_ACC_PATH, MOT_FRAMES_NUM)
        self.server2 = Server(2, "norway", "yolov5m",
                              GT_ACC_PATH, MOT_FRAMES_NUM)
        self.servers = OffloadingTargets([self.server1, self.server2])
        # other
        self.algorithm = algorithm
        self.steps_count = 0
        self.skipped_segment_count = 0
        self.segment_count = 0
        self.drain_mode = False
        self.log = LOG + algorithm + \
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".csv"
        self.training_log = LOG + algorithm + \
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + "-train.csv"
        self.remain_time = 0
        self.drain_step_count = 0

        # envs
        self.actions_mapping = self.__build_actions_mapping(SERVER_NUM + 1)
        self.action_space = spaces.Discrete(len(self.actions_mapping))
        self.observation_space = spaces.Dict({
            "past_bws_mean": spaces.Box(low=np.array([0, 0]), high=np.array(self.servers.get_max_bw())/BYTES_IN_MB, dtype=np.float32),
            "past_bws_std": spaces.Box(low=np.array([0, 0]), high=np.array(self.servers.get_max_bw())/BYTES_IN_MB, dtype=np.float32),
            "available_buffer_size": spaces.Box(low=-2 * BUFFER_SIZE/BYTES_IN_MB, high=2 * BUFFER_SIZE / BYTES_IN_MB, dtype=np.float32),
            "segment_size": spaces.Box(low=0, high=2.0 * BUFFER_SIZE / BYTES_IN_MB, dtype=np.float32),
            "past_segment_delay": spaces.Box(low=0, high=np.inf, dtype=np.float32)
        })
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

    def do_segment_analyzing(self, config):
        """sending and analyzing a segment.
        @params:
            config(Dict): resolution, framerate, quantizer, target
        @returns:
            results(List): List of metrics for each frame in the segment
            mAps(List): mAp of each frame
            analyzing_time(float): detector's processing time(milliseconds)
            encoding_time(int): time for encoding the frames into a segment(milliseconds)
            tranmission_time(float): transmission time for sending the segment from client to server  
            [bws_mean1, bws_mean2, bws_std1, bws_std2]: mean and std of all offloading targets bandwidth to send the segment
            segment_index(int): the index of the segment
            frames_id(List[int]): list of frames it contains
            segment_size(int): bytes of the segment
            local_processing_energy(float): if segment is processed locally
            tranmission_energy(float): if segment is sent to remote server
        """
        segment_index, frames_id, segment_size, encoding_time, resolution = self.client.get_segment()
        transmission_time = 0
        local_processing_energy, transmission_energy = energy_consuming(
            len(frames_id), config["resolution"], config["target"] == 0)
        bws1 = []
        bws2 = []
        if config["target"] != 0:
            server = self.servers.get_server_by_id(config["target"])
            bytes_to_send = segment_size
            results, mAps, analyzing_time, = server.analyze_video_segment(
                f"{self.client.tmp_segments}/{segment_index:06d}.avi", frames_id, config["resolution"])
            while bytes_to_send > 0:
                bws = self.servers.step_networks()
                bws1.append(bws[0])
                bws2.append(bws[1])
                throughputs = bws[config["target"] - 1]
                t = 1 if throughputs <= bytes_to_send else bytes_to_send / throughputs
                transmission_time += server.rtt + int(t * 1000)
                bytes_to_send -= throughputs
        else:
            results, mAps, analyzing_time = self.client.analyze_video_segment(
                f"{self.client.tmp_segments}/{segment_index:06d}.avi", frames_id, config["resolution"])
            bws1.append(self.servers.current_bws[0])
            bws2.append(self.servers.current_bws[1])
        return results, mAps, analyzing_time, encoding_time, transmission_time,\
            [np.mean(bws1), np.mean(bws2), np.std(bws1), np.std(bws2)], \
            segment_index, frames_id, segment_size, local_processing_energy, transmission_energy

    def update_state(self, config):
        """given an config(action) and update the environment.
        @params:
            config: Dict[resolution, framerate, quantizer, target]
        @returns:
            Dict:
                empty(bool): True if the client buffer is empty
                drain(bool):True if in draining_mode else False
                # results(List[float]): evaluated metrics of the segment
                mAps(float): mean mAp of all frames in the segment
                analyzing_time(int): millisecond that detectors takes to analyze the segment
                encoding_time(int): millisecond that gstreamer encodes the segment
                tranmission_time(int): millisecond to send the segment to the remote target
                capture_segment(int): num of segments that should captured from the stream while sending the current segment
                bws_mean1(float): mean bandwidth of remote target 1 to send the segment
                bws_mean2(float): mean bandwidth of remote target 2 to send the segment
                bws_std1(float): std of the bandwidth of remote target 1
                bws_std2(float): std of the bandwidth of remote target 2
        """
        if self.client.empty():
            self.remain_time = 0
            self.client.retrieve(config)
            self.segment_count += 1
            bws = [[] for _ in range(len(self.servers))]
            for _ in range(2):
                curr_bws = self.servers.step_networks()
                bws[0].append(curr_bws[0])
                bws[1].append(curr_bws[1])
            return {"empty": True, "drain_mode": self.drain_mode, "mAps": 0,
                    "analyzing_time": 0, "encoding_time": 0,
                    "transmission_time": 0, "capture_segments": 1,
                    "bws_mean1": np.mean(bws[0]), "bws_mean2": np.mean(bws[1]),
                    "bws_std1": np.std(bws[0]), "bws_std2": np.std(bws[1]), "segment_index": -1,
                    "frames_num": 0, "segment_size": 0,
                    "resolution": config["resolution"], "framerate": config["framerate"], "quantizer": config["quantizer"],
                    "target": config["target"], "remian_time": self.remain_time,
                    "available_buffer_size": self.client.get_buffer_vacancy(),
                    "processing_energy": 0, "transmission_energy": 0, "delay": 0}

        results, mAps, analyzing_time, encoding_time, tranmission_time, \
            [bws_mean1, bws_mean2, bws_std1, bws_std2], segment_index, frames_id, \
            segment_size, processing_energy, transmission_energy = self.do_segment_analyzing(
                config)
        self.remain_time += (tranmission_time / MILLS_PER_SECOND)
        capture_segments = int(
            self.remain_time // 2)
        self.remain_time -= capture_segments * 2
        for _ in range(capture_segments):
            self.client.retrieve(config, self.drain_mode)
            self.segment_count += 1
            self.skipped_segment_count += 1 if self.drain_mode else 0
            if not self.drain_mode and self.client.full():
                self.drain_mode = True
                self.remain_time = 0
        if self.drain_mode and self.client.empty():
            self.drain_mode = False
        # has removed "results" attribute in the return state
        return {"empty": False, "drain_mode": self.drain_mode, "mAps": np.mean(mAps),
                "analyzing_time": analyzing_time, "encoding_time": encoding_time,
                "transmission_time": tranmission_time, "capture_segments": capture_segments,
                "bws_mean1": bws_mean1, "bws_mean2": bws_mean2, "bws_std1": bws_std1,
                "bws_std2": bws_std2, "segment_index": segment_index,
                "frames_num": len(frames_id), "segment_size": segment_size,
                "resolution": config["resolution"], "framerate": config["framerate"], "quantizer": config["quantizer"],
                "target": config["target"], "remain_time": self.remain_time, "available_buffer_size": self.client.get_buffer_vacancy(),
                "processing_energy": processing_energy, "transmission_energy": transmission_energy,
                "delay": tranmission_time + encoding_time + analyzing_time}

    def _get_obs(self, state, config):
        """return observation."""
        obs = {"past_bws_mean": np.array([state["bws_mean1"]/BW_NORM, state["bws_mean2"]/BW_NORM]),
               "past_bws_std": np.array([state["bws_std1"]/BW_NORM, state["bws_std2"]/BW_NORM]),
               "available_buffer_size": np.array([self.client.get_buffer_vacancy() / BYTES_IN_MB]),
               "segment_size": np.array([state["segment_size"] / BYTES_IN_MB]),
               "past_segment_delay": np.array([state["delay"] / MILLS_PER_SECOND])}
        return obs

    def _get_reward(self, state, config):
        """given the state and action, return the reward."""
        # TODO(wuyuheng): tune reward
        # tradeoff (accuracy, energy, delay)
        alpha = 0.1
        beta = 0.1
        gamma = 0.1
        baseline = 3 * alpha
        if not state["drain_mode"] and self.drain_step_count > 0:
            self.drain_step_count = 0

        reward = state["mAps"] * config["framerate"] * alpha - state["delay"] / MILLS_PER_SECOND * beta - \
            (state["processing_energy"] +
             state["transmission_energy"]) / 1e10

        if state["drain_mode"]:
            self.drain_step_count += 1
            if self.drain_step_count == 1:
                return -5
            reward -= 0.5
        if config["target"] == 0:
            reward -= 0.5
        return reward

    def step(self, action):
        """given and perform an action and step to next state.
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
        config = self.actions_mapping[action]
        state = self.update_state(config)
        obs = self._get_obs(state, config)
        reward = self._get_reward(state, config)
        truncated = self.truncated()
        terminated = self.terminated()
        # logging
        log_info = dict(**state, cap_frames_num=self.client.cap_frames_num,
                        sent_frames_num=self.client.sent_frames_num)
        train_info = dict(**obs, reward=reward,
                          terminated=(terminated or truncated))
        log_info_df = pd.DataFrame([log_info])
        train_info_df = pd.DataFrame([train_info])
        log_info_df.to_csv(f"{self.log}", mode='a', header=None, index=False)
        train_info_df.to_csv(f"{self.training_log}",
                             mode='a', header=None, index=False)
        return obs, reward, terminated, truncated, state

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.servers.reset()
        self.steps_count = 0
        self.skipped_segment_count = 0
        self.segment_count = 0
        self.drain_mode = False
        obs = OrderedDict()
        obs["past_bws_mean"] = np.array([0.0, 0.0], dtype=np.float32)
        obs["past_bws_std"] = np.array([0.0, 0.0], dtype=np.float32)
        obs["available_buffer_size"] = np.array(
            [self.client.get_buffer_vacancy() / BYTES_IN_MB], dtype=np.float32)
        obs["segment_size"] = np.array([0], dtype=np.float32)
        obs["past_segment_delay"] = np.array([0], dtype=np.float32)
        return obs, {}

    def truncated(self):
        return self.steps_count > MAX_STEPS

    def terminated(self):
        return self.client.done()

    def close(self):
        return
