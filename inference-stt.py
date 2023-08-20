import os
import sys
import vits.inferencems as ims
import pyaudio
import whisper
import numpy as np
import time


def stopper(indata):
    return np.sum(np.square(indata)) / len(indata) # sqavg


def micinput(threshold):
    sampling = 16000
    maxdur = 3

    print(f"Beginning voice input sequence\nThis sequence ends after {maxdur} seconds of inactivity")
    frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling, input=True, frames_per_buffer=1024)
    elt = 0
    st = float(0)
    et = float(0)

    while True:
        st = time.time()
        data = stream.read(1024)
        et = time.time()
        frames.append(data)

        if elt >= maxdur:
            break
        elif stopper(np.frombuffer(b''.join(frames), dtype=np.int16)) < threshold:
            elt += et - st
        else:
            elt = 0

    print("Voice input sequence end")
    stream.stop_stream()
    stream.close()
    p.terminate()

    model = whisper.load_model("small")
    audio = whisper.pad_or_trim(np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 2 ** 15)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options).text

    print(f"Detected text : {result}")
    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference-stt.py {model_name} {model_step}")
        sys.exit(1)

    mic_threshold = 1000 # threshold to define empty input

    try:
        text = micinput(mic_threshold)
        ims.submodule(text)
    except:
        print("Error occurred")




if __name__ == "__main__":
    main()
