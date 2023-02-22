#!/bin/bash
width=(1920 1600 1280 960)
height=(1080 900 800 600)
frame_rate=(30 20 10 5)
constant_rate_factor=(18 21 23 26 28)
length = ${#width[@]}
for i in "${!width[@]}";
do
    for fr in ${frame_rate[@]}
    do
        for crf in ${constant_rate_factor[@]}
        do
            echo ${width[$i]}
            ../bin/ffmpeg -f image2 -r $fr -i MOT16-04/img1/%6d.jpg -vf scale=${width[$i]}:-1 -c:v libx264 -preset veryslow -crf $crf -tune zerolatancy -threads 0 videos/${width[$i]}_${fr}_${crf}.flv
            mkdir videos/${width[$i]}_${fr}_${crf}
            ffmpeg -i videos/${width[$i]}_${fr}_${crf}.flv videos/${width[$i]}_${fr}_${crf}/%6d.jpg
        done
    done
done