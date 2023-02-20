"""
dataset:
    unit_id,dtime,target,address,fetch_time,bytes_total,bytes_sec,objects,threads,requests,connections,reused_connections,lookups,request_total_time,request_min_time,request_avg_time,request_max_time,ttfb_total,ttfb_min,ttfb_avg,ttfb_max,lookup_total_time,lookup_min_time,lookup_avg_time,lookup_max_time,successes,failures
"""
import csv
import math
from tqdm import tqdm
import os
import argparse
import datetime

FILE_PATH = "./dataset/fcc/202201/curr_webget.csv"
COOKED_PATH = './dataset/fcc/202201cooked/'
MAHIMAHI_PATH = './dataset/fcc/202201traces/'
SIM_TRACE_PATH = './dataset/fcc/202201sim/'
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)
MILLISEC_IN_SEC = 1000.0
BYTES_PER_MTU = 1500.0

bw_measurements = {}


def cook_raw_data():
    # process curr_webget.csv and select throughput grouped by(uid, target)
    with open(FILE_PATH, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for line in tqdm(reader, desc="reading"):
            uid = line[0]
            target = line[2]
            throughput = line[6]
            id = (str(uid), target)
            if id in bw_measurements:
                bw_measurements[id].append(throughput)
            else:
                bw_measurements[id] = [throughput]

    for id in tqdm(bw_measurements, desc="save"):
        out_file = 'trace_' + \
            '_'.join(id).replace(':', '-').replace('/', '-')
        out_file = COOKED_PATH + out_file
        if not os.path.exists(COOKED_PATH):
            os.makedirs(COOKED_PATH)
            print(f"make new folder {COOKED_PATH}")
        with open(out_file, 'w') as f:
            for i in bw_measurements[id]:
                f.write(str(i) + '\n')


def convert2mahimahi():
    files = os.listdir(COOKED_PATH)
    for filename in tqdm(files, desc="converting"):
        out_file = MAHIMAHI_PATH + "mahi_" + filename
        if not os.path.exists(MAHIMAHI_PATH):
            os.makedirs(MAHIMAHI_PATH)
            print(f"make new folder {MAHIMAHI_PATH}")
        with open(COOKED_PATH + filename, 'r') as f, open(out_file, 'w') as mf:
            millsec_time = 0
            for throughput in f.readlines():
                mtu_pkts_per_millsec = float(
                    throughput) / MILLISEC_IN_SEC / BYTES_PER_MTU
                millsec_count = 0
                mtu_pkts_count = 0
                while True:
                    millsec_count += 1
                    millsec_time += 1
                    to_send = math.floor(
                        millsec_count * mtu_pkts_per_millsec) - mtu_pkts_count
                    mtu_pkts_count += to_send
                    for _ in range(to_send):
                        mf.write(str(millsec_time) + '\n')

                    if millsec_time >= MILLISEC_IN_SEC:
                        break


# def convert_sim_format():
#     """convert fcc to simulation trace format [timestamp, Mbits/sec]
#     """
#     sim_traces = {}
#     with open(FILE_PATH, "r") as f:
#         reader = csv.reader(f, delimiter=',')
#         next(reader)
#         for line in tqdm(reader, desc="reading"):
#             uid = line[0]
#             target = line[2]
#             try:
#                 dtime = (datetime.datetime.strptime(
#                     line[1], "%Y-%m-%d %H:%M:%S") - TIME_ORIGIN).total_seconds()
#             except:
#                 continue
#             else:
#                 throughput = line[6]
#                 id = (str(uid), target)
#                 if id in sim_traces:
#                     sim_traces[id].append([dtime, throughput])
#                 else:
#                     sim_traces[id] = [[dtime, throughput]]

#     for id in tqdm(sim_traces, desc="save"):
#         out_file = 'simtrace' + \
#             '_'.join(id).replace(':', '-').replace('/', '-')
#         out_file = SIM_TRACE_PATH + out_file
#         if not os.path.exists(SIM_TRACE_PATH):
#             os.makedirs(SIM_TRACE_PATH)
#             print(f"make new folder {SIM_TRACE_PATH}")
#         with open(out_file, 'w') as f:
#             for [timestamp, throughput] in sim_traces[id]:
#                 f.write(str(timestamp) + " " + str(throughput) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cook", action="store_true",
                        help="cook raw fcc trace")
    parser.add_argument("--convert", action="store_true",
                        help="convert cooked trace to Mahimahi format")
    # parser.add_argument('--cooksim', action="store_true",
    #                     help="convert raw fcc traces to simulation format")
    args = parser.parse_args()
    if args.cook:
        cook_raw_data()
    if args.convert:
        convert2mahimahi()
    # if args.cooksim:
    #     convert_sim_format()
    print("done")
