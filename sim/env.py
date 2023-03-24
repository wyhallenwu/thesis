import os
import numpy as np
from sim.detectors import DetrDetector, YoloDetector
from sim.util import GT, Evaluator
from sim.client import Client
import gym
from gym import spaces
from sim.server import Server
import itertools
from collections import OrderedDict
# action: [framerate, resolution, quantizer, offloading target]
# skip = [0, 1, 2, 4, 5] => fps = [30, 15, 10, 6, 5]
# timestamp is 2 seconds
MILLS_PER_SECOND = 1000
BUFFER_SIZE = 2000000  # bytes = 2Mb
MOT_DATASET_PATH = "MOT16-04"
MOT_FRAMES_NUM = 1050
TMP_PATH = "tmp"
TMP_FRAMES = TMP_PATH + "/frames"
TMP_CHUNKS = TMP_PATH + "/chunks"
GT_ACC_PATH = "acc"
FCC_PATH = "dataset/fcc/202201cooked"
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
QUANTIZER = [5, 15, 25, 35, 45]
SKIP = [0, 1, 2, 4, 5]
FRAMERATE = [30, 15, 10, 6, 5]
CLIENT_BUFFER_SIZE = 2000000
SERVER_NUM = 2
BANDWIDTH_BOUND = 1000000


class SimEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        # envs
        self.actions_mapping = self.__build_actions_mapping(SERVER_NUM + 1)
        # [skip(5), resolution(4), quantizer(5), links(n)] 5*4*5*n
        self.action_space = spaces.Discrete(len(self.actions_mapping))
        self.observation_space = spaces.Dict(
            {"past_average_bandwidth": spaces.Discrete(BANDWIDTH_BOUND),
             "client_buffer_size": spaces.Discrete(CLIENT_BUFFER_SIZE),
             "past_segment_size": spaces.Discrete(CLIENT_BUFFER_SIZE),
             "past_framerate": spaces.Discrete(5),
             "past_quantizer": spaces.Discrete(5),
             "past_resolution": spaces.Discrete(4)})
        # client
        self.client = Client(MOT_DATASET_PATH, TMP_PATH, BUFFER_SIZE)
        # remote servers
        self.server1 = Server(1, FCC_PATH, "detr", GT_ACC_PATH, MOT_FRAMES_NUM)
        self.server2 = Server(2, FCC_PATH, "yolov5m",
                              GT_ACC_PATH, MOT_FRAMES_NUM)
        self.servers = [self.server1, self.server2]
        # other
        self.captured_chunk_num = 0
        self.to_process_chunk_index = 0
        self.drain_status = False
        self.current_chunk_index = 0

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

    def update_state(self, config):
        # capture from video stream
        if not self.client.retrieve(config["framerate"]):
            # if buffer is full, drain buffer
            results, mAps, analyzing_times, encoding_times, tranmission_times,\
                chunk_counter, average_bws = self.do_drain_buffer(config)
            return {"event": "drain", "results": results, "mAps": mAps,
                    "analyzing_time": analyzing_times, "encoding_time": encoding_times,
                    "transmission_time": tranmission_times, "chunk_counter": chunk_counter, "average_bws": average_bws}
        else:
            results, mAps, analyzing_time, encoding_time, tranmission_time, average_bws = self.do_chunk_analyzing(
                config)
            time_consume = (analyzing_time + encoding_time +
                            tranmission_time) / MILLS_PER_SECOND
            capture_chunks = int(time_consume) // 2
            return {"event": "normal", "results": results, "mAps": mAps,
                    "analyzing_time": analyzing_time, "encoding_time": encoding_time,
                    "transmission_time": tranmission_time, "capture_chunks": capture_chunks, "average_bws": average_bws}

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
            average_bws(int): average bandwidth to send the current chunk
        """
        chunk_index, frames_id, chunk_size, encoding_time = self.client.get_chunk()
        transmission_time = 0
        if config["target"] != 0:
            bws = []
            server = self.servers[config["target"]]
            bytes_to_send = chunk_size
            results, mAps, analyzing_time = server.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
            while bytes_to_send > 0:
                bw, throughputs = server.step_network()
                bws.append(bw)
                t = 1 if throughputs <= bytes_to_send else bytes_to_send / throughputs
                transmission_time += server.rtt + int(t * 1000)
                bytes_to_send -= throughputs
            average_bws = np.mean(itertools.chain.from_iterable(bws))
        else:
            results, mAps, analyzing_time = self.client.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
            average_bws = 0
        return results, mAps, analyzing_time, encoding_time, transmission_time, int(average_bws)

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
        config = self.actions_mapping[action]
        info = self.update_state(config)
        obs = OrderedDict()
        obs["past_average_bandwidth"] = info["average_bws"]
        obs["client_buffer_vacancy"] = self.client.get_buffer_vacancy
        if info["event"] == "normal":
            pass
        else:
            pass
        obs = self.observation_space.sample()
        for k, v in obs.items():
            print(k, v)
        return self.observation_space.sample(), 1, True, False, {}

    def _get_obs(self, config):
        info = self.update_state(config)

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

    def do_drain_buffer(self, config):
        """when buffer is full, drain buffer and analyze all chunks in the buffer and then do next step.
        @param:
            config: Dict[resolution, framerate, quantizer, target]
        @return:
            results(List): wrapped result of earh chunk(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps(List): mAp of each chunk
            analyze_t(List): analyzing time of the chunks
            encoding_t(List): compressing time and encoding time of each chunk
            tranmission_time(int): time of the draining
            chunk_counter(int): chunks num drained
            average_bws(int): average bandwidth of the period to drain the buffer
        """

        # TODO: 改为处理每一个chunk，而不是一个buffer整个处理
        results, mAps, analyze_time, encoding_time = [], [], [], []
        transmission_time = 0
        chunk_counter = 0
        bytes_to_send = 0
        server = self.servers[config["target"]]
        rtt = server.rtt
        bws = []
        while self.client.buffer.empty():
            chunk_index, frames_id, chunk_size, processing_time = self.client.get_chunk()
            self.current_chunk_index = chunk_index
            result, mAp, analyzing_time = server.analyze_video_chunk(
                f"{self.client.tmp_chunks}/{chunk_index:06d}.avi", frames_id, config["resolution"])
            results.append(result)
            mAps.append(mAp)
            analyze_time.append(analyzing_time)
            encoding_time.append(processing_time)
            bytes_to_send += chunk_size
            chunk_counter += 1
        while bytes_to_send > 0:
            bw, throughputs = server.step_network()
            bws.append(bw)
            t = 1 if throughputs <= bytes_to_send else bytes_to_send / throughputs
            transmission_time += int(t * 1000)
            bytes_to_send -= throughputs
        average_bws = np.mean(itertools.chain.from_iterable(bws))
        return results, mAps, analyze_time, encoding_time, transmission_time + chunk_counter * rtt, chunk_counter, average_bws

    def reset(self, seed=None):
        super().reset(seed=seed)
        pass

    def clean_tmp_frames(self):
        os.system(f"rm -rf {self.client.tmp_frames}/*")
