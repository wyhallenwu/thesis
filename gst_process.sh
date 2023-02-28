#!/bin/bash
quantizer=(5 15 25 35 45)
frame_rate=(30 20 15 10 5)
width=(1920 1600 1280 960 640)
height=(1080 900 720 540 360)
for i in "${!width[@]}";
do
    for q in ${quantizer[@]}
    do
        for fr in ${frame_rate[@]}
        do
            echo ${width[$i]}x${height[$i]} ${q} ${fr}
            gst-launch-1.0 multifilesrc location=MOT16-04/img1/%06d.jpg start-index=1 stop-index=1050 caps="image/jpeg,framerate=${fr}/1" ! decodebin ! videoscale ! video/x-raw,width=${width[$i]},height=${height[$i]} !videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${q} tune=zerolatency threads=8 ! flvmux ! filesink location="gst_videos/${width[$i]}x${height[$i]}_${q}_${fr}.flv"
        done
    done
done