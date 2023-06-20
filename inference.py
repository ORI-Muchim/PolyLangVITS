# -*- coding: utf-8 -*-
import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py {model_name} {model_step}")
        sys.exit(1)

    model_name = sys.argv[1]
    model_step = sys.argv[2]

    command = ["python", "./vits/inferencems.py", model_name, model_step]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        print("Error occurred")

if __name__ == "__main__":
    main()
