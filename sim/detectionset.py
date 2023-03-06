from podm.metrics import BoundingBox
import os
from queue import Queue


class ClientBuffer():
    def __init__(self, dataset_path, buffer_size=5):
        self.dataset_path = dataset_path
        self.transmission_set = sorted(os.listdir(dataset_path))
        self.current_frame = 1
        self.buffer_size = buffer_size
        self.buffer = Queue(buffer_size)

    def __len__(self):
        return len(self.transmission_set)

    def push(self, frame_id, xmin, ymin, xmax, ymax, class_name, confidence, process_time):
        bbox = BoundingBox(frame_id, class_name, xmin,
                           ymin, xmax, ymax, confidence)
        self.pool.append(bbox)
        if self.process_time_map[frame_id] == None:
            self.process_time_map[frame_id] = process_time

    def retrieve(self, chunk_size, skip):
        counter = 0
        frame_set = []
        while counter < chunk_size:
            frame_set.append(self.current_frame)
            self.current_frame = (self.current_frame +
                                  skip) % len(self.transmission_set) + 1
            counter += 1
        return frame_set

    def get_video_chunk(self):
        return None if self.buffer.empty() else self.buffer.get()

    def get_frame_path(self, frame_index):
        return self.dataset_path + "/" + self.transmission_set[frame_index - 1]
