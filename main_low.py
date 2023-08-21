import os
import subprocess
import sys

#sys.argv[1] = ex) ko, ja, en, zh
#sys.argv[2] = ALL transcription saved in this txt. insert model name
#sys.argv[3] = sample rate

sys.argv[1] = sys.argv[1].lower()

os.chdir('./datasets')
subprocess.run(["python", "integral_low.py", sys.argv[1], sys.argv[2], sys.argv[3]])

os.chdir('../')
subprocess.run(["python", "get_pretrained_model.py", sys.argv[1], sys.argv[2]])

config_path = f"../datasets/{sys.argv[2]}.json"
model_name = sys.argv[2]

os.chdir('./vits')
subprocess.run(["python", "train_ms.py", "-c", config_path, "-m", model_name])

