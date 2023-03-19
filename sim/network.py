import os
import random


class NetworkSim():
    """
    NetworkSim is the simulated traces in the format: bytes_per_second
    """

    def __init__(self, id, traces_path) -> None:
        self.id = id
        self.time_step = 0
        self.bws = []
        self.init_fcc_traces(traces_path)
        self.bw_id = random.randint(0, len(self.bws)-1)
        self.bw = self.bws[self.bw_id]
        self.current_bw = None

    def next_bw(self):
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

    def step(self, elapsed_time):
        return [self.next_bw() for _ in range(elapsed_time)]


class Networks():
    def __init__(self, num, traces_path) -> None:
        self.networks = [NetworkSim(id, traces_path) for i in range(num)]
        self.current_bws = None
        self.current_throughputs = None
        self.links = num
        self.timestamp = 0
        self.step()

    def step(self):
        """one timestamp elapsed. default step is 2 seconds
        @return:
            bws: bandwidth for each second in the past step
            throughputs: available bytes in the past step
        """
        bws = [net.step() for net in self.networks]
        throughputs = [sum(bw) for bw in bws]
        self.current_bws = bws
        self.current_throughputs = throughputs
        self.timestamp += 1
        return bws, throughputs

    def get_curr_throughput(self, id):
        assert id < self.links, "out of range id [Networks get_curr_throughput()]"
        return self.current_throughputs[id]

    def get_curr_bw(self, id):
        assert id < self.links, "out of range id [Networks get_curr_bw()]"
        return self.current_bws[id]

    def test(self):
        print("current bandwidth of each link: ", self.current_bws)
        print("current throughputs of each link: ", self.current_throughputs)
        print("current timestamp: ", self.timestamp)


if __name__ == "__main__":
    networks = Networks(2, traces_path="dataset/fcc/202201cooked/")
    networks.test()
