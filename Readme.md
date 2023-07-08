- # PolyLangVITS

Multilingual Speech Synthesis System Using [VITS](https://github.com/jaywalnut310/vits)


## Table of Contents 
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Prepare_Datasets](#Prepare_Datasets)
- [Usage](#usage)
- [Inference](#inference)
- [References](#References)

## Prerequisites
- A Windows/Linux system with a minimum of `16GB` RAM.
- A GPU with at least `12GB` of VRAM.
- Python == 3.8
- Anaconda installed.
- PyTorch installed.
- CUDA 11.x installed.
- Zlib DLL installed.

Pytorch install command:
```sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

CUDA 11.7 install:
`https://developer.nvidia.com/cuda-11-7-0-download-archive`

Zlib DLL install:
`https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows`

Install pyopenjtalk Manually:
`pip install -U pyopenjtalk --no-build-isolation`

---

## Installation 
1. **Create an Anaconda environment:**

```sh
conda create -n polylangvits python=3.8
```

2. **Activate the environment:**

```sh
conda activate polylangvits
```

3. **Clone this repository to your local machine:**

```sh
git clone https://github.com/ORI-Muchim/PolyLangVITS.git
```

4. **Navigate to the cloned directory:**

```sh
cd PolyLangVITS
```

5. **Install the necessary dependencies:**

```sh
pip install -r requirements.txt
```

---

## Prepare_Datasets

Place the audio files as follows. 

.mp3 or .wav files are okay. 

You must write '[language code]' on the back of the speaker folder.

```
PolyLangVITS
├────datasets
│       ├───speaker0[KO]
│       │   ├────1.mp3
│       │   └────1.wav
│       └───speaker1[JA]
│       │    ├───1.mp3
│       │    └───1.wav
│       ├───speaker2[EN]
│       │   ├────1.mp3
│       │   └────1.wav
│       ├───speaker3[ZH]
│       │   ├────1.mp3
│       │   └────1.wav
│       ├integral.py
│       └integral_low.py
│
├────vits
├────get_pretrained_model.py
├────inference.py
├────main_low.py
├────main_resume.py
├────main.py
├────Readme.md
└────requirements.txt
```

This is just an example, and it's okay to add more speakers.

---

## Usage

To start this tool, use the following command, replacing {language}, {model_name}, and {sample_rate} with your respective values:

```sh
python main.py {language} {model_name} {sample_rate}
```

For those with low specifications(VRAM < 12GB), please use this code:

```sh
python main_low.py {language} {model_name} {sample_rate}
```

If the data configuration is complete and you want to resume training, enter this code:

```sh
python main_resume.py {model_name}
```

---
## Inference

After the model has been trained, you can generate predictions by using the following command, replacing {model_name} and {model_step} with your respective values:

```sh
python inference.py {model_name} {model_step}
```

---
## References

For more information, please refer to the following repositories: 
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits.git) 
- [CjangCjengh/vits](https://github.com/CjangCjengh/vits.git)
- [Kyubyong/g2pK](https://github.com/Kyubyong/g2pK)
- [tenebo/g2pk2](https://github.com/tenebo/g2pk2)