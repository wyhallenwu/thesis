import cv2
from util.utils import get_mill_sec_timestamp


class DetectorYolo():
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.model = self.init_yolo_net(self.config, self.weights)

    def init_yolo_net(self, config, weights):
        """
        create and initialize a yolov3 model
        @params:
            config: yolov3 config file
            weights: yolov3 weights file
        """
        model = cv2.dnn_DetectionModel(config, weights)
        model.setInputParams(size=(250, 250), scale=1/255.0)
        model.setInputSwapRB(True)
        return model

    def detect(self, frame):
        start_time = get_mill_sec_timestamp()
        classes, confidences, boxes = self.model.detect(frame)
        end_time = get_mill_sec_timestamp()
        # processing time as a part of observation
        processing_time = end_time - start_time
        shape = frame.shape
        h = shape[0]
        w = shape[1]
        return [[h, w], processing_time, classes, confidences, boxes]
