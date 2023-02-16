import cv2
import time
import subprocess
from queue import Queue
from util.utils import get_mill_sec_timestamp

rtmp_url = "rtmp://localhost:1935/test"

"""
stream trace format:
id, time, frame, dection_result
"""


def format_recv_frame(frame_idx, start_time, frame_data):
    # milliseconds since the start of the epoch
    millsec_since_epoch = get_mill_sec_timestamp() - start_time
    return [frame_idx, millsec_since_epoch, frame_data, None]


def server():
    cap = cv2.VideoCapture(rtmp_url)
    while True:
        _, frame = cap.read()
        if frame is None:
            break

    cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    server()
