import csv
import time
from tqdm import tqdm

FILE_PATH = "./dataset/202212/curr_webget.csv"
OUTPUT_PATH = './dataset/cooked/'

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
        out_file = OUTPUT_PATH + out_file
        with open(out_file, 'w') as f:
            for i in bw_measurements[id]:
                f.write(str(i) + '\n')


def convert2mahimahi():
    pass


if __name__ == '__main__':
    cook_raw_data()
