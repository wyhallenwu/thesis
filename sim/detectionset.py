from podm.metrics import BoundingBox


class DetectionSet():
    def __init__(self):
        self.pool = []
        self.process_time_map = {}

    def __len__(self):
        return len(self.pool)

    def push(self, frame_id, xmin, ymin, xmax, ymax, class_name, confidence, process_time):
        bbox = BoundingBox(frame_id, class_name, xmin,
                           ymin, xmax, ymax, confidence)
        self.pool.append(bbox)
        if self.process_time_map[frame_id] == None:
            self.process_time_map[frame_id] = process_time
