from detector import DetectorYolo
from queue import Queue
from rl import A3C
import zerorpc


class ProfilingModule():
    def __init__(self, detect_config, model_weights, frame_buffer_length):
        self.ending_flag = False
        self.next_frame_idx = 0
        self.detector = DetectorYolo(detect_config, model_weights)
        self.buffer = Queue(frame_buffer_length)
        self.adpator = A3C()
        self.in_stream = None  # TODO

    def in_buffer(self, frame):
        self.next_frame_idx += 1
        if self.buffer.full():
            return False
        self.buffer.put(frame)
        return True

    def out_buffer(self):
        # TODO: using asychronous queue
        pass

    def profile(self):

        frame = self.buffer.get()
        result = self.detector.detect(frame)
        # TODO: format a episode

    def transmission_end_rpc(self):
        """rpc call provided for the client side.Once the 
        video transmission is finished, the client will call
        this function to notify the status. 
        """
        if self.buffer.empty():
            self.ending_flag = True
            return "end"
        return "running"
