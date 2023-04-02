import os
import random
import numpy as np

TRACES = {"fcc": "dataset/fcc/202201cooked", "norway": "dataset/norway"}


class NetworkSim():
    """
    NetworkSim is the simulated traces in the format: bytes_per_second
    """

    def __init__(self, traces="fcc") -> None:
        self.traces_path = TRACES[traces]
        self.time_step = 0
        self.bws = []
        if traces == "fcc":
            self.init_fcc_traces(self.traces_path)
            self.bw_id = random.randint(0, len(self.bws)-1)
            self.bw = self.bws[self.bw_id]
        elif traces == "norway":
            self.init_norway3G_trace(self.traces_path)
            self.bw = self.bws
        self.bw = np.roll(self.bw, random.randint(
            0, len(self.bw) - 1)).tolist()
        self.current_bw = 0

    def next_bw(self):
        self.time_step = (self.time_step + 1) % len(self.bw)
        self.current_bw = self.bw[self.time_step]
        return self.current_bw

    def get_current_bw(self):
        return self.current_bw

    def get_max_bw(self):
        return max(self.bw)

    def init_fcc_traces(self, fcc_cooked_path):
        files = os.listdir(fcc_cooked_path)
        for file in files:
            trace = []
            with open(f"{fcc_cooked_path}/{file}", 'r') as f:
                lines = f.readlines()
                for bw in lines:
                    trace.append(int(bw))
            self.bws.append(trace)

    def init_norway3G_trace(self, traces_path):
        files = os.listdir(traces_path)
        for file in files:
            with open(traces_path + "/" + file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    bw = int(line[4]) / int(line[5]) * 1000  # bytes per second
                    self.bws.append(int(bw / 4))

    def step(self):
        return self.next_bw()

    def reset(self):
        self.time_step = 0
        self.current_bw = 0

    def test(self):
        for _ in range(10):
            print(self.step())


if __name__ == "__main__":
    network = NetworkSim(traces="norway")
    network.test()
