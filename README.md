## push to server

ffmpeg -re -i input.mp4 -vcodec libx264 -c copy -f flv rtmp://100.64.0.1:1935/live/test

casva

## schedule

- [ ] server side
- [ ] reading and detecting from rtmp stream
- [ ] iot side, stream configuration(frame rate, resotion, server selection)
