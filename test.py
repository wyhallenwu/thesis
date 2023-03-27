import subprocess
import os
# f"gst-launch-1.0 multifilesrc location={self.tmp_frames}/%06d.jpg start-index=1
# caps='image/jpeg,framerate={config['framerate']}/1' ! decodebin ! videoscale !
# video/x-raw,width=${config['resolution'][0]},height=${config['resolution'][1]} !videoconvert !
# x264enc pass=5 speed-preset=1 quantizer=${config['quantizer']} tune=zerolatency threads=8 ! avimux !
# filesink location='{self.tmp_chunks}/{chunk_index:06d}.avi'")

os.system(
    "gst-launch-1.0 multifilesrc location=MOT16-04/img1/%06d.jpg start-index=1 caps=\"image/jpeg,framerate=30/1\" ! decodebin ! videoscale ! video/x-raw,width=1920,height=1080 ! videoconvert ! x264enc pass=5 speed-preset=1 quantizer=20 tune=zerolatency threads=8 ! avimux ! filesink location=\"test.avi\"")
