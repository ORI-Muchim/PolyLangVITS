import os
import sys
import pyaudio
import whisper
import numpy as np
import time


def stopper(indata):
    return np.sum(np.square(indata)) / len(indata) # sqavg


def micinput(threshold):
    sampling = 16000
    maxdur = 3
    print(f"Beginning voice input sequence\n** This sequence ends after {maxdur} seconds of inactivity **")
    maxdur = float(maxdur)

    frames = []
    subf = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling, input=True, frames_per_buffer=1024)
    elt: float = 0.0

    while True:
        st: float = time.time()
        data = stream.read(1024)
        et: float = time.time()
        frames.append(data)
        subf.append(data)

        if elt >= maxdur:
            break
        elif stopper(np.frombuffer(b''.join(subf), dtype=np.int16)) < threshold:
            elt += et - st
        else:
            elt: float = 0.0
        subf = []

    print("Voice input sequence end")
    stream.stop_stream()
    stream.close()
    p.terminate()

    model = whisper.load_model("large-v2")
    audio = whisper.pad_or_trim(np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 2 ** 15)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options).text

    print(f"Detected text : {result}")
    return result


def main():
    if len(sys.argv) == 3:
        mic_threshold = 500  # threshold to define empty input
        text = micinput(mic_threshold)
        model_name = sys.argv[1]
        model_step = sys.argv[2]

        command = f"python ./vits/inferencems.py {model_name} {model_step} {text}"
        return_code = os.system(command)

        if return_code != 0:
            print("Error occurred")
    elif len(sys.argv) > 3:
        model_name = sys.argv[1]
        model_step = sys.argv[2]
        text = sys.argv[3]

        if len(sys.argv) > 4: # in case when a user passes a sentence level text without quotation marks
            for i in range(4, len(sys.argv)+1):
                text += " " + sys.argv[i]

        command = f"python ./vits/inferencems.py {model_name} {model_step} {text}"
        return_code = os.system(command)

        if return_code != 0:
            print("Error occurred")
    else:
        print("Usage: python inference-stt.py {model_name} {model_step}\nOR\npython inference-stt.py {model_name} {model_step} {text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
