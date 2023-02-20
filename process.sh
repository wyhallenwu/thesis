#!/bin/bash
width=(1920 1280 854 640 426)
height=(1080 720 480 320 240)
frame_rate=(30 15 10 5 3 2 1)
constant_rate_factor=(0 10 20 30 40)
length = ${#width[@]}
for i in "${!width[@]}";
do
    for fr in ${frame_rate[@]}
    do
        for crf in ${constant_rate_factor[@]}
        do
            ffmpeg -r $fr -f image2 -i ~/MOT16-04/img1/%6d.jpg -c:v libx264 -crf $crf -s ${width[$i]}x${height[$i]} ${width[$i]}x${height[$i]}_$fr_$crf.flv
        done
    done
done

