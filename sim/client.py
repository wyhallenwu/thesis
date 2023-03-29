import os
import cv2
import time
from sim.detectors import YoloDetector
from collections import deque
from sim.util import Evaluator
import subprocess
"""
config: [width, height, quantizer, framerate]
"""
# each segment is set to be 2 seconds(30fps default)
DEFAULT_FRAMES_NUM = 30 * 2
# fps [30, 15, 10, 6, 5] => [0, 1, 2, 4, 5]
SKIP_MAPPING = {30: 0, 15: 1, 10: 2, 6: 4, 5: 5}


class Client():
    def __init__(self, dataset_path, gt_path, tmp_dir="tmp", buffer_size=2000000) -> None:
        self.dataset_path = dataset_path
        self.gt_path = gt_path
        self.buffer_size = buffer_size
        self.used_buffer = 0
        self.dataset = Dataset(dataset_path)
        self.buffer = deque()
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_chunks = tmp_dir + "/chunks"
        self.tmp_chunk_num = 0
        self.detector = YoloDetector("yolov5n")
        self.evaluator = Evaluator("acc", "yolov5n", 1050)
        self.rtt = 0
        self.traverse_count = 0
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        subprocess.run(f"rm -rf {self.tmp_chunks}/*", shell=True)
        print("BUILD CLIENT DONE.")

    def reset(self):
        self.tmp_chunk_num = 0
        self.used_buffer = 0
        self.dataset.current_frame_id = 1
        self.traverse_count = 0
        self.buffer.clear()
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        subprocess.run(f"rm -rf {self.tmp_chunks}/*", shell=True)

    def get_chunk(self):
        """get video chunk_index and frames_id from buffer.
        @return:
            chunk_index: the index to the latest chunk
            frames_id: wrapped frames of the current
            chunk_size: bytes of chunk
            encoding_time: encoding process time
        """
        chunk_index, frames_id, chunk_size, encoding_time, resolution = self.buffer.popleft()
        self.used_buffer -= chunk_size
        return chunk_index, frames_id, chunk_size, encoding_time, resolution

    def gstreamer(self, config, chunk_index):
        """process images with gstreamer and return the processing time"""
        start = time.time()
        res = subprocess.run(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps=\"image/jpeg,framerate={config['framerate']}/1\" ! decodebin ! videoscale ! video/x-raw,width={config['resolution'][0]},height={config['resolution'][1]} ! videoconvert ! x264enc pass=5 speed-preset=1 quantizer={config['quantizer']} tune=zerolatency threads=8 ! avimux ! filesink location=\"{self.tmp_chunks}/{chunk_index:06d}.avi\"", shell=True)
        res.check_returncode()
        end = time.time()
        return round((end - start) * 1000, 3)

    def process_video(self, frames_id, config, chunk_index):
        """process_video using gstreamer to compress the frames into avi following the configuration(resolution, quantizer)
        @params:
            frames_id: List[frame_id:int]
            config:Dict[resolution, framerate, quantizer, target]
            chunk_index: current chunk index
        @return:
            chunk_size, processing_time
        """
        for id, frame_id in enumerate(frames_id):
            subprocess.run(
                f"cp {self.gt_path}/{config['resolution'][0]}x{config['resolution'][1]}/{frame_id:06d}.jpg {self.tmp_frames}/{(id+1):06d}.jpg", shell=True)
        gst_time = self.gstreamer(config, chunk_index)
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        return os.path.getsize(f"{self.tmp_chunks}/{chunk_index:06d}.avi"), gst_time

    def retrieve(self, config, drain_mode=False):
        # TODO: add resolution in consideration
        """retrieve frames at every interval skip. if buffer is full, abandon the capture
        @params:
            config: Dict[resolution, framerate, quantizer, target]
        """
        skip = SKIP_MAPPING[config["framerate"]]
        frames_id = self.capture(DEFAULT_FRAMES_NUM / (skip + 1),
                                 skip)  # default frames in each segment is 60
        if drain_mode:
            return
        if not self.full():
            self.tmp_chunk_num += 1
            chunk_size, encoding_time = self.process_video(
                frames_id, config, self.tmp_chunk_num)
            self.used_buffer += chunk_size
            self.buffer.append([self.tmp_chunk_num, frames_id,
                                chunk_size, encoding_time, config["resolution"]])

    def capture(self, chunk_size, skip):
        """retrieve chunk_size frames per second with the interval of skip.
        @params:
            chunk_size: num of frames to capture
            skip: interval => framerate
        @return:
            frames: list of frame id
        """
        counter = 0
        frames_id = []
        while counter < chunk_size:
            frames_id.append(self.dataset.current_frame_id)
            self.traverse_count += 1 if self.dataset.current_frame_id + \
                skip >= len(self.dataset) else 0
            self.dataset.current_frame_id = (self.dataset.current_frame_id +
                                             skip) % len(self.dataset) + 1
            counter += 1
        return frames_id

    def analyze_video_chunk(self, chunk_filename, frames_id, resolution):
        """current video chunk is processed by local device.
        @param:
            chunk_filename: path to the video chunk 
            frames_id: List[int] index of each frame
            resolution: [width, height]
        @return:
            results: wrapped result of earh frame(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps: mAp of each frame
            processing_time: process time of the whole chunk
            processing_energy: j/chunk energy cost of processing one chunk
            transmission_energy: j/chunk energy cost of transmitting one chunk
        """
        bboxes, processing_time = self.detector.detect_video_chunk(
            chunk_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, f"{resolution[0]}x{resolution[1]}", frame_id)
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def full(self):
        return self.used_buffer >= self.buffer_size

    def get_buffer_vacancy(self):
        return self.buffer_size - self.used_buffer

    def empty(self):
        return self.get_buffer_vacancy() == 0

    def done(self):
        return self.traverse_count >= 3


class Dataset():
    def __init__(self, dataset_path) -> None:
        self.files = sorted(os.listdir(dataset_path))
        self.current_frame_id = 1

    def __len__(self):
        return len(self.files)
