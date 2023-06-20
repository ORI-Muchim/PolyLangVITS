import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from scipy.io.wavfile import write
import soundfile as sf
from text2phonemesequence import Text2PhonemeSequence
import sys


def get_inputs(text, model, tokenizer_xphonebert):
    phones = model.infer_sentence(text)
    tokenized_text = tokenizer_xphonebert(phones)
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    input_ids = torch.LongTensor(input_ids).cuda()
    attention_mask = torch.LongTensor(attention_mask).cuda()
    return input_ids, attention_mask

hps = utils.get_hparams_from_file(f"./models/{sys.argv[1]}/config.json")

tokenizer_xphonebert = AutoTokenizer.from_pretrained(hps.bert)

# Load Text2PhonemeSequence
model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
net_g = SynthesizerTrn(
    hps.bert,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(f"./models/{sys.argv[1]}/G_{sys.argv[2]}.pth", net_g, None)

output_dir = f'./vitsoutput/{sys.argv[1]}'
os.makedirs(output_dir, exist_ok=True)

n_speakers = hps.data.n_speakers

text = '''
"[KO]가장 밝게 빛나는 순간은 주위의 모든 것이 가장 어두울 때이다.[KO]"
'''

for idx in range(5):
    sid = torch.LongTensor([idx]).cuda()
    stn_tst, attention_mask = get_inputs(text, model, tokenizer_xphonebert)
    print(text)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        attention_mask = attention_mask.cuda().unsqueeze(0)
        audio = net_g.infer(x_tst, attention_mask, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(f'{output_dir}/output{idx}.wav', hps.data.sampling_rate, audio)
    print(f'{output_dir}/output{idx}.wav Generated!')