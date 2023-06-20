import os
import subprocess
import sys

config_path = f"../datasets/{sys.argv[1]}.json"
model_name = sys.argv[1]

os.chdir('./vits')
command = ["python", "train_ms.py", "-c", config_path, "-m", model_name]

subprocess.run(command)