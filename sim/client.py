import numpy as np
from sim.buffers import ClientBuffer
import os
from util import logger
import cv2
import time

"""
config: [width, height, quantizer, framerate]
"""
# each segment is set to be 2 seconds(30fps default)
DEFAULT_FRAMES_NUM = 30 * 2
# fps [30, 15, 10, 6, 5] => [0, 1, 2, 4, 5]
SKIP_MAPPING = {30: 0, 15: 1, 10: 2, 6: 4, 5: 5}


class Client():
    def __init__(self, dataset_path, tmp_dir="tmp") -> None:
        self.dataset_path = dataset_path
        self.client_buffer = ClientBuffer(dataset_path)
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_chunks = tmp_dir + "/chunks"
        self.tmp_chunk_counter = 0
        self.logger = logger(f"{self.tmp_dir}/train.log")

    def full(self):
        return self.client_buffer.full()

    def empty(self):
        return self.client_buffer.empty()

    def get_frames_buffer(self):
        return self.client_buffer.get_video_chunk()

    def gstreamer(self, config):
        """process images with gstreamer and return the processing time"""
        start = time.time()
        os.system(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps='image/jpeg,framerate={config[3]}/1' ! decodebin ! videoscale ! video/x-raw,width=${config[0]},height=${config[1]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${config[2]} tune=zerolatency threads=8 ! flvmux ! filesink location='{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv'")
        end = time.time()
        return round((end - start) * 1000, 3)

    def process_video(self, frames_id, config):
        """process_video using gstreamer to compress the frames into flv following the configuration(resolution, quantizer)
        @params:
            frames_id: List(frame_id)
            config:[resolution, quantizer]
        @return:
            chunk_index, chunk_size, processing_time
        """
        log_info = ""
        # clean the tmp_frames and copy the chunk images to the tmp
        os.system(f"rm -rf {self.tmp_frames}/*")
        for id, frame_id in enumerate(frames_id):
            img = cv2.imread(f"{self.dataset_path}/{frame_id}.jpg")
            cv2.imwrite(f"{self.tmp_frames}/{id+1:06d}.jpg", img)
            log_info += f"{frame_id} "
        gst_time = self.gstreamer(config)
        self.tmp_chunk_counter += 1
        log_info += f"{self.tmp_chunk_counter} {config[0]}x{config[1]} {gst_time}"
        self.logger(log_info)
        return self.tmp_chunk_counter, os.path.getsize(f"{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv"), gst_time

    def retrieve(self, skip):
        """retrieve frames at every interval skip. if buffer is full, abandon the capture
        @params:
            skip(int): [0, 1, 2, 4, 5] => fps=[30, 15, 10, 6, 5]
        @return:
            bool: return True if buffer is not full else False
        """
        frames_id = self.capture(DEFAULT_FRAMES_NUM / (skip + 1),
                                 skip)  # default frames in each segment is 60
        if not self.client_buffer.buffer.full():
            self.client_buffer.buffer.put(frames_id)
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
            frames_id.append(self.current_frame)
            self.current_frame = (self.current_frame +
                                  skip) % len(self.client_buffer) + 1
            counter += 1
        return frames_id

    def step(self, config):
        """step action for each timestamp.
        @params:
            config: List[width, height, quantizer, framerate]
        """
        skip = SKIP_MAPPING[config[3]]
        if self.full():
            pass


# class Streams():
#     def __init__(self, dataset_path, name) -> None:
#         self.dataset_path = dataset_path
#         self.frames = sorted(os.listdir(self.dataset_path))
#         self.name = name

#     def __len__(self):
#         return len(self.frames)
