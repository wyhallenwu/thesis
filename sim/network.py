import os
import random


class NetworkSim():
    """
    network trace is in the format of [bytes_per_millisecond]
    trace file can be cooked fcc traces. convert bytes_per_sec
    each timestep is one second
    """

    def __init__(self, id, fcc_path) -> None:
        self.id = id
        self.time_step = 0
        self.bws = []
        self.init_fcc_traces(fcc_path)
        self.bw_id = self.bws[random.randint(0, len(self.bws)-1)]
        self.bw = self.bws[self.bw_id]
        self.current_bw = None

    def next_timestep(self):
        self.time_step = (self.time_step + 1) % len(self.bw)
        self.current_bw = self.bw[self.time_step]
        return self.current_bw

    def get_id(self):
        return self.id

    def init_fcc_traces(self, fcc_cooked_path):
        files = os.listdir(fcc_cooked_path)
        for file in files:
            trace = []
            with open(f"{fcc_cooked_path}/{file}", 'r') as f:
                lines = f.readlines()
                for bw in lines:
                    trace.append(int(bw))
            self.bws.append(trace)


class Networks():
    def __init__(self, num, traces_path) -> None:
        self.networks = [NetworkSim(id, traces_path) for i in range(num)]

    def next_bws(self):
        bws = []
        for net in self.networks:
            bws.append(net.next_timestep())
        return bws
