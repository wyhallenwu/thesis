# import subprocess

# input_file = "/home/wuyuheng/Downloads/bw.avi"
# server = "rtmp://100.64.0.1:1935/test"

# command = [
#     'ffmpeg',
#     '-re', '-i',
#     input_file,
#     '-vcodec', 'libx264',
#     '-acodec', 'aac',
#     '-f', 'flv',
#     server,
# ]

# subprocess.run(command)

from util.utils import get_config
print(get_config())
