# README

code for undergraduate thesis

## structure

```
./
├── MOT16-04             # MOT16-04 dataset
├── README.md
├── acc                  # pre-analyzed result
├── fig                  # figures that used in the thesis
├── gt                   # resized MOT16-04 with different [width, height]
├── log                  # training log
├── main.py              # run this file
├── model                # saving trained model
├── motivation           #
├── preprocess           # code for preprocess and motivation
├── requirements.yaml
├── sim                  # simulation environment
├── tmp                  # temporary files during training
└── traces               # network bandwidth trace
```

## datasets

### network traces source

```
# fcc dataset: https://www.fcc.gov/oet/mba/raw-data-releases
curr_webget.csv: unit_id,dtime,target,address,fetch_time,bytes_total,bytes_sec,objects,threads,requests,connections,reused_connections,lookups,request_total_time,request_min_time,request_avg_time,request_max_time,ttfb_total,ttfb_min,ttfb_avg,ttfb_max,lookup_total_time,lookup_min_time,lookup_avg_time,lookup_max_time,successes,failures


# belgium 4G/LTE bandwidth: https://users.ugent.be/~jvdrhoof/dataset-4g/

Number of milliseconds since epoch;
Number of milliseconds since start of experiment;
GPS latitude in decimal degrees;
GPS longitude in decimal degrees;
Number of bytes received since last datapoint;
Number of milliseconds since last datapoint.

An example looks like this:
1453121790686 39287 51.0386528885778 3.73220785642186 2493100 1000
1453121791687 40288 51.0386532820580 3.73222847166054 2763504 1001
1453121792686 41287 51.0386553301824 3.73225142858955 2728896 0999
1453121793686 42287 51.0386567711171 3.73227435543338 1954600 1000

# norway HSDPA bandwidth log: http://skuld.cs.umass.edu/traces/mmsys/2013/pathbandwidth/
using the first (bus) log
```

### MOT16-04 source

```
MOT16-04: https://motchallenge.net/data/MOT16/
```

## before training pipeline

如果是国内用户可能需要科学上网，因为`detector.py`中的 model(detr, yolov5)需要从 huggingface 和 pytorch hub 中获取。

```bash
conda env create --name thesis --file requirements.txt
# resize MOT16-04 frames
python sim/preprocess.py --src=MOT16-04/img1 --saving=gt
# using different object detection models to get groundtruth detection result
python sim/preprocess.py --gt_path=gt --saving=acc --model=detr
python sim/preprocess.py --gt_path=gt --saving=acc --model=yolov5m
python sim/preprocess.py --gt_path=gt --saving=acc --model=yolov5n
```

## start training

```bash
# ppo
python main.py --algo=ppo
# a2c
python main.py --algo=a2c
```

# experiments result
