f = open('../genshin_datasets/raiden/metadata.csv', 'r', encoding='utf-8').read().split('\n')

l = []
import os
import wave
import contextlib

c = 0
for i in f:
    p, t = i.split('|')

    with contextlib.closing(wave.open(p, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        c+=duration

print('time = ', c)