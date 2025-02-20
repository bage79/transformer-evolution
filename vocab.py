import argparse
import csv
import os
import sys

import pandas as pd
import sentencepiece as spm

from config import is_mac_or_pycharm


def build_corpus(downloaded_file, corpus_file):
    """ pretrain corpus 생성 (.csv -> .txt) """
    csv.field_size_limit(sys.maxsize)
    SEPARATOR = u"\u241D"
    LINE_SEPARATOR = "\n\n\n\n"  # TODO: '\n\n' 으로 해도 될 듯
    df = pd.read_csv(downloaded_file, sep=SEPARATOR, engine="python")

    with open(corpus_file, "w") as f:
        for index, row in df.iterrows():
            f.write(row["text"].lower())
            f.write(LINE_SEPARATOR)
            print(f"build corpus ... {index + 1} / {len(df)}", end="\r")
    print()
    return corpus_file


def build_vocab(corpus_file, prefix, vocab_size):
    """ vocab 생성 (.txt -> .model, .vocab) """
    spm.SentencePieceTrainer.train(
        f"--input={corpus_file} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +
        " --unk_id=1 --unk_piece=[UNK]" +
        " --bos_id=2 --bos_piece=[BOS]" +
        " --eos_id=3 --eos_piece=[EOS]" +
        " --user_defined_symbols=[SEP],[CLS],[MASK]"  # TODO: --control_symbols 를 사용하면 안되나? 차이는?
        " --character_coverage=0.98"
        " --hard_vocab_limit=false")


def load_vocab(file):
    """ vocab 로드 """
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data" if not is_mac_or_pycharm() else "data_sample", type=str, required=False,
                        help="a data directory which have downloaded, corpus text, vocab files.")
    parser.add_argument("--vocab_size", default=8000, type=int, required=False,
                        help="max vocab size")
    args = parser.parse_args()
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    print(args)

    if not os.path.exists("data"):
        os.makedirs("data")

    for downloaded_file in os.listdir(args.data_dir):
        if downloaded_file.endswith('.csv'):
            downloaded_file = os.path.join(args.data_dir, downloaded_file)
            corpus_file = downloaded_file.replace('.csv', '.txt')
            vocab_file = downloaded_file.replace('.csv', '.vocab')
            vocab_model_file = downloaded_file.replace('.csv', '.model')

            prefix = downloaded_file.replace('.csv', '')
            if downloaded_file.endswith('.csv') and not os.path.isfile(corpus_file):
                build_corpus(downloaded_file, corpus_file)
            if not os.path.exists(vocab_file) or not os.path.exists(vocab_model_file):
                build_vocab(corpus_file, prefix, args.vocab_size)
