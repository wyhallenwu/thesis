# README

tips for undergraduate thesis code.

## dataset

```
fcc dataset: https://www.fcc.gov/oet/mba/raw-data-releases
curr_webget.csv: unit_id,dtime,target,address,fetch_time,bytes_total,bytes_sec,objects,threads,requests,connections,reused_connections,lookups,request_total_time,request_min_time,request_avg_time,request_max_time,ttfb_total,ttfb_min,ttfb_avg,ttfb_max,lookup_total_time,lookup_min_time,lookup_avg_time,lookup_max_time,successes,failures


belgium 4G/LTE bandwidth: https://users.ugent.be/~jvdrhoof/dataset-4g/

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

norway HSDPA bandwidth log: http://skuld.cs.umass.edu/traces/mmsys/2013/pathbandwidth/
using the first one (bus) log
```

## simulation traces

```
fork code from pensieve:
run the make_traces.py in sim/ via python2 to get simulation traces
```

## environment

```
./configure --prefix=/home/shdx/disk2/wuyuheng/thesis/dev/ --enable-shared --enable-gpl --enable-libx264 --enable-cuda-nvcc --enable-nonfree --disable-x86asm
```

## push to server

ffmpeg -re -i input.mp4 -vcodec libx264 -c copy -f flv rtmp://100.64.0.1:1935/live/test

## setup
```
python sim/preprocess.py --src=MOT16-04/img1 --saving=gt
python sim/preprocess.py --gt_path=gt --saving=acc --model=detr
```

## schedule

- [ ] detect all preprocessed video clips with pytorch yolov5 and split them into seperate jpg files and recored related result
