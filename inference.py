# -*- coding: utf-8 -*-
import sys
import subprocess
import vits.utils

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py {model_name}")
        sys.exit(1)

    hps = vits.utils.get_hparams()

    model_name = sys.argv[1]
    model_step = vits.utils.load_checkpoint(vits.utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"))

    command = f"python ./vits/inferencems.py {model_name} {model_step}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8'), end='')
        
    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        print("Error occurred")

if __name__ == "__main__":
    main()

