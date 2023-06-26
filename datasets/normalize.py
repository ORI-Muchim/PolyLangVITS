import os
import subprocess
import shutil

directory = "./datasets/zss[ZH]/wavs"

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        original_file = os.path.join(directory, filename)
        temp_file = os.path.join(directory, "temp_" + filename)

        subprocess.run(['ffmpeg', '-i', original_file, '-af', 'volume=0dB:precision=double', temp_file])

        shutil.move(temp_file, original_file)
