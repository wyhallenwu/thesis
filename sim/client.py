import os
import cv2
import time
from sim.detectors import YoloDetector
from queue import Queue
"""
config: [width, height, quantizer, framerate]
"""
# each segment is set to be 2 seconds(30fps default)
DEFAULT_FRAMES_NUM = 30 * 2
# fps [30, 15, 10, 6, 5] => [0, 1, 2, 4, 5]
SKIP_MAPPING = {30: 0, 15: 1, 10: 2, 6: 4, 5: 5}

BUFFER_SIZE = 10


class Client():
    def __init__(self, dataset_path, tmp_dir="tmp") -> None:
        self.dataset_path = dataset_path
        self.dataset = Dataset(dataset_path)
        self.buffer = Queue(BUFFER_SIZE)
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_chunks = tmp_dir + "/chunks"
        self.tmp_chunk_num = 0
        self.detector = YoloDetector("yolov5n")
        print("BUILD CLIENT DONE.")

    def get_chunk_filename(self, chunk_index: int):
        return self.tmp_chunks + f"/{chunk_index:06d}.avi"

    def gstreamer(self, config):
        """process images with gstreamer and return the processing time"""
        start = time.time()
        os.system(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps='image/jpeg,framerate={config[3]}/1' ! decodebin ! videoscale ! video/x-raw,width=${config[0]},height=${config[1]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${config[2]} tune=zerolatency threads=8 ! avimux ! filesink location='{self.tmp_chunks}/{self.tmp_chunk_counter:06d}.avi'")
        end = time.time()
        return round((end - start) * 1000, 3)

    def process_video(self, frames_id, config):
        """process_video using gstreamer to compress the frames into avi following the configuration(resolution, quantizer)
        @params:
            frames_id: List[frame_id:int]
            config:[width, height, quantizer]
        @return:
            chunk_index, chunk_size, processing_time
        """
        # clean the tmp_frames and copy the chunk images to the tmp
        os.system(f"rm -rf {self.tmp_frames}/*")
        for frame_id in frames_id:
            os.system(
                f"cp {self.dataset_path}/{frame_id:06d}.jpg {self.tmp_frames}/{frame_id:06d}.jpg")
        gst_time = self.gstreamer(config)
        self.tmp_chunk_num += 1
        return self.tmp_chunk_num, os.path.getsize(f"{self.tmp_chunks}/{self.tmp_chunk_num:06d}.avi"), gst_time

    def retrieve(self, framerate):
        """retrieve frames at every interval skip. if buffer is full, abandon the capture
        @params:
            skip(int): [0, 1, 2, 4, 5] => fps=[30, 15, 10, 6, 5]
        @return:
            bool: return True if buffer is not full else False
        """
        skip = SKIP_MAPPING[framerate]
        frames_id = self.capture(DEFAULT_FRAMES_NUM / (skip + 1),
                                 skip)  # default frames in each segment is 60
        if not self.buffer.full():
            self.buffer.put(frames_id)
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

    # TODO: modify
    def step(self, config):
        """step action for each timestamp.
        @params:
            config: Dict[[width, height], quantizer, framerate, target]
        """
        skip = SKIP_MAPPING[config[2]]
        full_flag = self.retrieve(skip)
        if not full_flag:
            return "buffer full"
        else:
            frames_id = self.get_frames_id()
            self.process_video()


class Dataset():
    def __init__(self, dataset_path) -> None:
        self.files = sorted(os.listdir(dataset_path))
        self.current_frame_id = 1

    def __len__(self):
        return len(self.files)
