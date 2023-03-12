import numpy as np
from sim.buffers import ClientBuffer
import os
from util import logger
import cv2
import time


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

    def process_video(self, frames, config):
        """process_video using gstreamer to compress the frames into flv following the configuration(resolution, quantizer)
        @params:
            frames: List(frame_id)
            config:[resolution, quantizer]
        @return:
            chunk_index, chunk_size
        """
        log_info = ""
        os.system(f"rm -rf {self.tmp_frames}/*")
        for id, frame_id in enumerate(frames):
            img = cv2.imread(f"{self.dataset_path}/{frame_id}.jpg")
            cv2.imwrite(f"{self.tmp_frames}/{id+1:06d}.jpg", img)
            log_info += f"{frame_id} "
        self.tmp_chunk_counter += 1
        start_time = time.time()
        os.system(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps='image/jpeg,framerate={len(frame_chunk)}/1' ! decodebin ! videoscale ! video/x-raw,width=${config[0][0]},height=${config[0][1]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${config[1]} tune=zerolatency threads=8 ! flvmux ! filesink location='{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv'")
        end_time = time.time()
        gst_time = round((end_time - start_time) * 1000, 3)
        log_info += f"{self.tmp_chunk_counter} {config[0][0]}x{config[0][1]} {gst_time}"
        self.logger(log_info)
        return self.tmp_chunk_counter, os.path.getsize(f"{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv")

    def retrieve(self, skip):
        """retrieve frames at every interval skip. if buffer is full, abandon the capture
        @params:
            skip(int): [0, 1, 2, 4, 5] => fps=[30, 15, 10, 6, 5]
        @return:
            bool: return True if buffer is not full else False
        """
        frames = self.capture(30 / (skip + 1), skip)
        if not self.client_buffer.buffer.full():
            self.client_buffer.buffer.put(frames)
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
        frames = []
        while counter < chunk_size:
            frames.append(self.current_frame)
            self.current_frame = (self.current_frame +
                                  skip) % len(self.client_buffer) + 1
            counter += 1
        return frames
