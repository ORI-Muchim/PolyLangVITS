import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py {model_name} {model_step}")
        sys.exit(1)

    model_name = sys.argv[1]
    model_step = sys.argv[2]

    command = f"python ./vits/inferencems.py {model_name} {model_step}"

    return_code = os.system(command)

    if return_code != 0:
        print("Error occurred")

if __name__ == "__main__":
    main()
