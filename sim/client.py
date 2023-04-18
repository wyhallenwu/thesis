import os
import time
from sim.detectors import YoloDetector
from collections import deque
from sim.util import Evaluator
import subprocess

# =======================hyperparameter====================
# each segment is set to be 2 seconds(30fps default)
DEFAULT_FRAMES_NUM = 30 * 2
# fps [30, 15, 10, 6, 5] => [0, 1, 2, 4, 5]
SKIP_MAPPING = {30: 0, 15: 1, 10: 2, 6: 4, 5: 5}
# =========================================================


class Client():
    """Client is a simulated smart camera with preliminary computing resource to handle object detection task.\
        Default deployed model is yolov5n."""

    def __init__(self, dataset_path, gt_path, tmp_dir="tmp", buffer_size=2000000) -> None:
        self.dataset_path = dataset_path
        self.gt_path = gt_path
        self.buffer_size = buffer_size
        self.used_buffer = 0
        self.dataset = Dataset(dataset_path)
        self.buffer = deque()
        self.tmp_dir = tmp_dir  # folder for the tmp compressed videos
        self.tmp_frames = tmp_dir + "/frames"
        self.tmp_segments = tmp_dir + "/segments"
        self.tmp_segment_num = 0
        self.detector = YoloDetector("yolov5n")
        self.evaluator = Evaluator("acc", "yolov5n", 1050)
        self.rtt = 0
        self.cap_frames_num = 0
        self.sent_frames_num = 0
        if not os.path.exists(self.tmp_frames):
            os.makedirs(self.tmp_frames)
        if not os.path.exists(self.tmp_segments):
            os.makedirs(self.tmp_segments)
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        subprocess.run(f"rm -rf {self.tmp_segments}/*", shell=True)
        print("BUILD CLIENT DONE.")

    def reset(self):
        self.tmp_segment_num = 0
        self.used_buffer = 0
        self.dataset.current_frame_id = 1
        self.cap_frames_num = 0
        self.sent_frames_num = 0
        self.buffer.clear()
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        subprocess.run(f"rm -rf {self.tmp_segments}/*", shell=True)

    def get_segment(self):
        """pop video segment in the buffer.
        @return:
            segment_index: the index to the latest segment
            frames_id: wrapped frames of the current
            segment_size: bytes of segment
            encoding_time: encoding process time
        """
        segment_index, frames_id, segment_size, encoding_time, resolution = self.buffer.popleft()
        self.used_buffer -= int(segment_size)
        return segment_index, frames_id, segment_size, encoding_time, resolution

    def gstreamer(self, config, segment_index):
        """helper function to wrap frames into video segments with gstreamer and return the processing time(ms)"""
        start = time.time()
        res = subprocess.run(
            f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1 caps=\"image/jpeg,framerate={config['framerate']}/1\" ! decodebin ! videoscale ! video/x-raw,width={config['resolution'][0]},height={config['resolution'][1]} ! videoconvert ! x264enc pass=5 speed-preset=1 quantizer={config['quantizer']} tune=zerolatency threads=8 ! avimux ! filesink location=\"{self.tmp_segments}/{segment_index:06d}.avi\"", shell=True)
        res.check_returncode()
        end = time.time()
        return round((end - start) * 1000, 3)

    def process_video(self, frames_id, config, segment_index):
        """wrap frames into video segment according to the configuration(resolution, quantizer)
        @params:
            frames_id: List[frame_id:int]
            config:Dict[resolution, framerate, quantizer, target]
            segment_index: current segment index
        @return:
            segment_size, processing_time
        """
        for id, frame_id in enumerate(frames_id):
            subprocess.run(
                f"cp {self.gt_path}/{config['resolution'][0]}x{config['resolution'][1]}/{frame_id:06d}.jpg {self.tmp_frames}/{(id+1):06d}.jpg", shell=True)
        gst_time = self.gstreamer(config, segment_index)
        subprocess.run(f"rm -rf {self.tmp_frames}/*", shell=True)
        return os.path.getsize(f"{self.tmp_segments}/{segment_index:06d}.avi"), gst_time

    def retrieve(self, config, drain_mode=False):
        """retrieve frames at every interval skip. if buffer is full or in drain buffer mode, drop the captured frames.
        @params:
            config: Dict[resolution, framerate, quantizer, target]
        """
        skip = SKIP_MAPPING[config["framerate"]]
        frames_id = self.capture(DEFAULT_FRAMES_NUM / (skip + 1),
                                 skip)  # default frames in each segment is 60
        if drain_mode:
            return
        self.tmp_segment_num += 1
        self.sent_frames_num += len(frames_id)
        segment_size, encoding_time = self.process_video(
            frames_id, config, self.tmp_segment_num)
        self.used_buffer += segment_size
        self.buffer.append([self.tmp_segment_num, frames_id,
                            segment_size, encoding_time, config["resolution"]])

    def capture(self, frames_num: int, skip: int):
        """helper function to capture num frames per time interval.
        @params:
            segment_size(int): num of frames to capture
            skip(int): interval => framerate
        @return:
            frames(List[int]): list of frame id
        """
        counter = 0
        frames_id = []
        while counter < frames_num:
            frames_id.append(self.dataset.current_frame_id)
            self.dataset.current_frame_id = (self.dataset.current_frame_id +
                                             skip) % len(self.dataset) + 1
            counter += 1
        self.cap_frames_num += frames_num
        return frames_id

    def analyze_video_segment(self, segment_filename, frames_id, resolution):
        """current video segment is processed by local device.
        @param:
            segment_filename(str): path to the video segment 
            frames_id(List[int]):  index of each frame
            resolution(List[int]): [width, height]
        @return:
            results(List[float]): result of earh frame(ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection)
            mAps[List(float)]: mAp of each frame
            processing_time(float): process time of the whole segment
        """
        bboxes, processing_time = self.detector.analyze_single_segment(
            segment_filename, frames_id)
        mAps = []
        results = []
        for boxes, frame_id in zip(bboxes, frames_id):
            result, mAp = self.evaluator.evaluate(
                boxes, resolution, frame_id)
            results.append(result)
            mAps.append(mAp)
        return results, mAps, processing_time

    def full(self):
        return self.used_buffer >= self.buffer_size

    def get_buffer_vacancy(self):
        return self.buffer_size - self.used_buffer

    def empty(self):
        return self.used_buffer == 0

    def done(self):
        """after captured 10000 frames, the episode is done."""
        return self.cap_frames_num >= 10000


class Dataset():
    """Dataset stores the frames dataset which will be used to simulated the live video stream.\
        Default is MOT16-04."""

    def __init__(self, dataset_path) -> None:
        self.files = sorted(os.listdir(dataset_path))
        self.current_frame_id = 1

    def __len__(self):
        return len(self.files)
