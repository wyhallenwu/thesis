import subprocess
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sim.server import Server
from tqdm import tqdm
PATH = "/"
ACC = "acc/"
RESOLUTION = [[1920, 1080], [1600, 900], [1280, 720], [960, 540]]
FRAMERATE = [30, 15, 10, 5]
QUANTIZER = [18, 23, 28, 33, 38, 43]


def gstreamer(resolution, framerate, quantizer):
    res = subprocess.run(
        f"gst-launch-1.0 multifilesrc location=MOT16-04/img1/%06d.jpg start-index=1 caps=\"image/jpeg,framerate={framerate}/1\" ! decodebin ! videoscale ! video/x-raw,width={resolution[0]},height={resolution[1]} ! videoconvert ! x264enc pass=5 speed-preset=1 quantizer={quantizer} tune=zerolatency threads=8 ! avimux ! filesink location=\"{PATH}/{resolution[0]}-{framerate}-{quantizer}.avi\"", shell=True)
    res.check_returncode()


def observation():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    os.system(f"rm -rf {PATH}*")
    for resolution in RESOLUTION:
        for framerate in FRAMERATE:
            for quantizer in QUANTIZER:
                gstreamer(resolution, framerate, quantizer)


def plot_fr_q():
    files = os.listdir(PATH)
    fr_q_groups = {'resolution': [], 'fr-q': [], 'size': []}
    for file in files:
        rs, fr, q = file[:-4].split('-')
        if int(q) not in QUANTIZER:
            continue
        fr_q_groups['resolution'].append(int(rs))
        fr_q_groups['fr-q'].append(f"{int(fr):02d}-{int(q):02d}")
        fr_q_groups['size'].append(int(os.path.getsize(PATH+file)))
    fr_q_groups = pd.DataFrame.from_dict(
        fr_q_groups)
    fr_q_groups.sort_values(["fr-q", "resolution"], inplace=True)
    fr_q_groups.to_csv('fig/fr_q.csv', index=False, columns=fr_q_groups.keys())
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 8))
    plt.xticks(rotation=60)
    plt.xlabel("framerate-quantizer")
    plt.ylabel("segment size/bytes")
    sns.lineplot(data=fr_q_groups, x='fr-q', y='size',
                 hue='resolution', style='resolution', linewidth=2.5, dashes=True,
                 ax=ax, palette="RdYlBu_r")
    f.savefig("fig/fr_q.pdf", dpi=1200)


def plot_acc_rs():
    """acc-config"""
    detr_detector = Server(1, "norway", "detr", "acc", 1050)
    frames_id = list(range(1, 1051))
    resolution = {1920: [1920, 1080], 1600: [
        1600, 900], 1280: [1280, 720], 960: [960, 540]}
    result = {'rs': [], 'fr': [], 'q': [], 'mAp': []}
    for file in tqdm(os.listdir(PATH)):
        rs, fr, q = file[:-4].split('-')
        if int(q) not in QUANTIZER:
            continue
        res = resolution[int(rs)]
        _, mAps, _ = detr_detector.analyze_video_chunk(
            PATH+file, frames_id, res)
        result['rs'].append(rs)
        result['fr'].append(f"{int(fr):02d}")
        result['q'].append(f"{int(q):02d}")
        result['mAp'].append(round(np.mean(mAps), 3))
        print(file)
    result = pd.DataFrame.from_dict(result)
    result.sort_values(
        ['rs', 'fr', 'q'], inplace=True)
    result.to_csv('acc_config.csv', index=False, columns=result.keys())
    # plot
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 8))
    plt.xticks(rotation=60)
    plt.xlabel("framerate-quantizer")
    plt.ylabel("mAp")
    result['fr-q'] = result['fr'] + "-" + result['q']
    sns.lineplot(data=result, x='fr-q', y='mAp', hue='rs', style='rs', linewidth=2.5, dashes=True,
                 ax=ax, palette='RdYlBu_r')
    f.savefig('fig/fr_q_rs.pdf', dpi=1200)


def plot_trace(traces_path: str):
    files = os.listdir(traces_path)
    bws = []
    for file in files:
        with open(traces_path + "/" + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                bw = int(line[4]) / int(line[5]) * 1000  # bytes per second
                bws.append(bw/1000000)
    bws = np.array(bws)
    f, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=bws, ax=ax)
    f.savefig('fig/norway_trace.pdf', dpi=1200)


if __name__ == '__main__':
    # observation()
    # plot_fr_q()
    # plot_acc_rs()
    plot_trace("dataset/norway")
