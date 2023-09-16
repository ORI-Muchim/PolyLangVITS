import os
import subprocess
import sys

#sys.argv[1] = ex) ko, ja, en, zh
#sys.argv[2] = ALL transcription saved in this txt. insert model name
#sys.argv[3] = sample rate

if len(sys.argv) < 4:
    print("Usage: script_name <language> <model_name> <sample_rate>")
    sys.exit(1)

language = sys.argv[1].lower()
model_name = sys.argv[2]
sample_rate = int(sys.argv[3])

os.chdir('./datasets')
subprocess.run(["python", "integral_low.py", language, model_name, str(sample_rate)])

os.chdir('../')
if sample_rate == 44100:
    subprocess.run(["python", "get_pretrained_model.py", language, model_name])

config_path = f"../datasets/{model_name}.json"

os.chdir('./vits')
subprocess.run(["python", "train_ms.py", "-c", config_path, "-m", model_name])
