import argparse
import text
from utils import load_filepaths_and_text
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
import re

# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")

# Load Text2PhonemeSequence
ph_ko = Text2PhonemeSequence(language='kor', is_cuda=True)
ph_ja = Text2PhonemeSequence(language='jpn', is_cuda=True)
ph_en = Text2PhonemeSequence(language='eng-us', is_cuda=True)
ph_zh = Text2PhonemeSequence(language='zho-s', is_cuda=True)

def ipa_cleaners(text):
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: ph_zh.infer_sentence(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: ph_ja.infer_sentence(x.group(1))+' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: ph_ko.infer_sentence(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: ph_en.infer_sentence(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=2, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--language", default="eng-us")
  parser.add_argument("--cuda", default=False, action='store_true')
  parser.add_argument("--pretrained_g2p_model", default="charsiu/g2p_multilingual_byT5_small_100")
  parser.add_argument("--tokenizer", default="google/byt5-small")
  parser.add_argument("--batch_size", default=64)
  args = parser.parse_args()

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text = ipa_cleaners(original_text)
      filepaths_and_text[i][args.text_index] = cleaned_text
      # Print the cleaned text
      print("Original Text:", original_text)
      print("Cleaned Text:", cleaned_text)
      print("\n")

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
