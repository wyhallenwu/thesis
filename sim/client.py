import numpy as np
from detectionset import ClientBuffer
import os
from util import logger
import cv2
import time


class Client():
    def __init__(self, dataset_path, tmp_dir="tmp") -> None:
        self.client_buffer = ClientBuffer(dataset_path)
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_chunks = tmp_dir + "/chunks"
        self.tmp_chunk_counter = 0
        self.logger = logger(f"{self.tmp_dir}/train.log")

    def process_video(self, frame_chunk, config):
        """process_video using gstreamer to compress the frames into flv following the configuration(resolution, quantizer)
        @params:
            video_chunk: List(frame_id)
            config:[resolution, quantizer]
        @return:
            chunk_index, chunk_size
        """
        log_info = ""
        os.system(f"rm -rf {self.tmp_frames}/*")
        for id, frame_path in enumerate(frame_chunk):
            img = cv2.imread(frame_path)
            cv2.imwrite(f"{self.tmp_frames}/{id+1:06d}.jpg", img)
            log_info += f"{frame_path[:-4]} "
        self.tmp_chunk_counter += 1
        start_time = time.time()
        os.system(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps='image/jpeg,framerate={len(frame_chunk)}/1' ! decodebin ! videoscale ! video/x-raw,width=${config[0][0]},height=${config[0][1]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${config[1]} tune=zerolatency threads=8 ! flvmux ! filesink location='{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv'")
        end_time = time.time()
        gst_time = round((end_time - start_time) * 1000, 3)
        log_info += f"{self.tmp_chunk_counter} {config[0][0]}x{config[0][1]} {gst_time}"
        self.logger(log_info)
        return self.tmp_chunk_counter, os.path.getsize(f"{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.flv")

    def capture(self, skip):
        """capture(): capture frames at every interval skip.
        @params:
            skip(int): [0, 1, 2, 4, 5] => fps=[30, 15, 10, 6, 5]
        @return:
            bool: return True if buffer is not full else False
        """
        if not self.client_buffer.buffer.full():
            self.client_buffer.retrieve(30 / (skip+1), skip)
            return True
        return False
