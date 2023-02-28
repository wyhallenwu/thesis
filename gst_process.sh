#!/bin/bash
quantizer=(10 20 30 40 50)
frame_rate=(30 20 15 10 5)
for q in ${quantizer[@]}
do
    for fr in ${frame_rate[@]}
    do
        gst-launch-1.0 multifilesrc location=MOT16-04/img1/%06d.jpg start-index=1 stop-index=1050 caps="image/jpeg,framerate=${fr}/1" ! decodebin ! videoconvert ! x264enc pass=5 speed-preset=1 quantizer=${q} tune=zerolatency threads=4 ! flvmux ! filesink location="test${q}_${fr}.flv"
    done
done
