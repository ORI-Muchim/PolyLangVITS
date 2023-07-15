import os
import random
import re

original_filename = "./datasets/finetuning_val.txt.cleaned"

new_filename = "./datasets/finetuning_val.txt.cleaned2"

with open(original_filename, 'r', encoding='utf-8') as infile:
    with open(new_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            match = re.search(r'\|\d+\|', line)
            if match:
                random_number = random.randint(0, 499)
                new_line = line.replace(match.group(), f"|{random_number}|")
                outfile.write(new_line)
            else:
                outfile.write(line)
