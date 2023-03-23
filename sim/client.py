import os
import cv2
import time
from sim.detectors import YoloDetector
from collections import deque
from queue import Queue
"""
config: [width, height, quantizer, framerate]
"""
# each segment is set to be 2 seconds(30fps default)
DEFAULT_FRAMES_NUM = 30 * 2
# fps [30, 15, 10, 6, 5] => [0, 1, 2, 4, 5]
SKIP_MAPPING = {30: 0, 15: 1, 10: 2, 6: 4, 5: 5}


class Client():
    def __init__(self, dataset_path, tmp_dir="tmp", buffer_size=2000000) -> None:
        self.dataset_path = dataset_path
        self.buffer_size = buffer_size
        self.used_buffer = 0
        self.dataset = Dataset(dataset_path)
        self.buffer = Queue()
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_chunks = tmp_dir + "/chunks"
        self.tmp_chunk_num = 0
        self.detector = YoloDetector("yolov5n")
        self.rtt = 0
        print("BUILD CLIENT DONE.")

    def get_chunk(self):
        """get video chunk_index and frames_id from buffer.
        @return:
            chunk_index: the index to the latest chunk
            frames_id: wrapped frames of the current
            chunk_size: bytes of chunk
            encoding_time: encoding process time
        """
        chunk_index, frames_id, chunk_size, encoding_time = self.buffer.get()
        return chunk_index, frames_id, chunk_size, encoding_time

    def gstreamer(self, config, chunk_index):
        """process images with gstreamer and return the processing time"""
        start = time.time()
        os.system(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps='image/jpeg,framerate={config['framerate']}/1' ! decodebin ! videoscale ! video/x-raw,width=${config['resolution'][0]},height=${config['resolution'][1]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${config['quantizer']} tune=zerolatency threads=8 ! avimux ! filesink location='{self.tmp_chunks}/{chunk_index:06d}.avi'")
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
        # clean the tmp_frames and copy the chunk images to the tmp
        os.system(f"rm -rf {self.tmp_frames}/*")
        for frame_id in frames_id:
            os.system(
                f"cp {self.dataset_path}/{frame_id:06d}.jpg {self.tmp_frames}/{frame_id:06d}.jpg")
        gst_time = self.gstreamer(config, chunk_index)
        return os.path.getsize(f"{self.tmp_chunks}/{chunk_index:06d}.avi"), gst_time

    def retrieve(self, config):
        """retrieve frames at every interval skip. if buffer is full, abandon the capture
        @params:
            config: Dict[resolution, framerate, quantizer, target]
        @return:
            bool: return True if buffer is not full else False
        """
        skip = SKIP_MAPPING[config["framerate"]]
        frames_id = self.capture(DEFAULT_FRAMES_NUM / (skip + 1),
                                 skip)  # default frames in each segment is 60
        if not self.full():
            self.tmp_chunk_num += 1
            chunk_size, encoding_time = self.process_video(
                frames_id, config, self.tmp_chunk_num)
            self.used_buffer += chunk_size
            self.buffer.put([self.tmp_chunk_num, frames_id,
                            chunk_size, encoding_time])
            return True
        return False

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
            self.dataset.current_frame_id = (self.dataset.current_frame_id +
                                             skip) % len(self.dataset.current_frame_id) + 1
            counter += 1
        return frames_id

    def _get_obs(self):
        return self.used_buffer

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
        """
        bboxes, processing_time = self.detector.detect_video_chunk(
            chunk_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, f"{resolution[0]}x{resolution[1]}", f"{frame_id:06d}")
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def full(self):
        return self.used_buffer <= self.buffer_size


class Dataset():
    def __init__(self, dataset_path) -> None:
        self.files = sorted(os.listdir(dataset_path))
        self.current_frame_id = 1

    def __len__(self):
        return len(self.files)
