import os
import math
from tqdm import tqdm

DATASET_PATH = "./dataset/norway/raw/bus/"
MAHIMAHI_PATH = "./dataset/norway/traces/bus/"

MTU_SIZE = 1500.0


def convert2mahimahi():
    files = os.listdir(DATASET_PATH)
    if not os.path.exists(MAHIMAHI_PATH):
        os.makedirs(MAHIMAHI_PATH)
        print(f"make new folder {MAHIMAHI_PATH}")
    for file_name in tqdm(files, desc="converting"):
        out_file = MAHIMAHI_PATH + "mahi_" + file_name
        time_idx = 0
        with open(DATASET_PATH + file_name, 'r') as f, open(out_file, 'w') as mf:
            for line in f.readlines():
                parsed = line.split()
                bytes_num = float(parsed[4])
                time_duration_millsec = float(parsed[5])
                mtu_pkts_per_millsec = bytes_num / time_duration_millsec / MTU_SIZE

                time_count = 0
                pkt_count = 0
                while True:
                    time_count += 1
                    time_idx += 1
                    to_send = math.floor(
                        time_count * mtu_pkts_per_millsec) - pkt_count
                    for _ in range(int(to_send)):
                        mf.write(str(time_idx) + "\n")

                    pkt_count += to_send
                    if time_count >= time_duration_millsec:
                        break


if __name__ == '__main__':
    convert2mahimahi()
    print("done")
