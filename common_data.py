import sys, os, argparse, datetime, time, re, collections
import wget
from tqdm import tqdm, trange
import pandas as pd
import json

from vocab import load_vocab, build_corpus


def prepare_train(args, vocab, infile, outfile):
    """ train data 준비 """
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = { "id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"] }
            f.write(json.dumps(instance))
            f.write("\n")
            print(f"build {outfile} {index + 1} / {len(df)}", end="\r")


def download_data(args):
    """ 데이터 다운로드 """
    print("download data/ratings_train.txt")
    filename = wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", "data")
    print()
    print("download data/ratings_test.txt")
    filename = wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", "data")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prepare", type=str, required=False,
                        help="동작모드 입니다. download: 학습할 데이터 다운로드, prepare: 학습할 데이터셋 생성")
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")
    
    if args.mode == "download":
        download_data(args)
    elif args.mode == "prepare":
        vocab = load_vocab("kowiki.model")
        args.corpus = "data/kowiki.txt"
        if not os.path.isfile(args.corpus):
            build_corpus("data/kowiki.csv", args.corpus)
        if not os.path.isfile("data/ratings_train.json"):
            prepare_train(args, vocab, "data/ratings_train.txt", "data/ratings_train.json")
        if not os.path.isfile("data/ratings_test.json"):
            prepare_train(args, vocab, "data/ratings_test.txt", "data/ratings_test.json")
    else:
        print(f"지원하지 않는 모드 입니다. {args.mode}\n- downlaod: 학습할 데이터 다운로드\n- preapre: 학습할 데이터셋 생성")
        exit(1)

