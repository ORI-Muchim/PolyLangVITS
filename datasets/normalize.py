import os
import subprocess
import shutil

# 정규화하려는 디렉토리
directory = "./datasets/zss[ZH]/wavs"

# 모든 .wav 파일을 순회
for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        original_file = os.path.join(directory, filename)
        temp_file = os.path.join(directory, "temp_" + filename)

        # ffmpeg를 사용하여 각 파일을 노멀라이즈하고 임시 파일에 저장
        subprocess.run(['ffmpeg', '-i', original_file, '-af', 'volume=0dB:precision=double', temp_file])

        # 임시 파일을 원본 파일로 덮어씀
        shutil.move(temp_file, original_file)
