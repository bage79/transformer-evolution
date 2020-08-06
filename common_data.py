import argparse
import json
import os

import pandas as pd
import wget

from config import IS_MAC
from vocab import load_vocab, build_corpus


def prepare_train(args, vocab, infile, outfile):
    """ train data 준비 """
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = {"id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"]}
            f.write(json.dumps(instance, ensure_ascii=False))
            f.write("\n")
            print(f"build {outfile} {index + 1} / {len(df)}", end="\r")


def download_data(data_dir):
    """ 데이터 다운로드 """
    print("download data/ratings_train.txt")
    wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", data_dir)
    print()
    print("download data/ratings_test.txt")
    wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", data_dir)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data" if not IS_MAC else "data_sample", type=str, required=False,
                        help="a data directory which have downloaded, corpus text, vocab files.")
    args = parser.parse_args()
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    print(args)

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.isfile(os.path.join(args.data_dir, "ratings_train.json")) or not os.path.isfile(os.path.join(args.data_dir, "ratings_test.json")):
        download_data(args.data_dir)

    vocab = load_vocab(os.path.join(args.data_dir, "kowiki.model"))
    if not os.path.isfile(os.path.join(args.data_dir, "kowiki.json")):
        prepare_train(args, vocab, os.path.join(args.data_dir, "kowiki.csv"), os.path.join(args.data_dir, "kowiki.json"))
    if not os.path.isfile(os.path.join(args.data_dir, "ratings_train.json")):
        prepare_train(args, vocab, os.path.join(args.data_dir, "ratings_train.txt"), os.path.join(args.data_dir, "ratings_train.json"))
    if not os.path.isfile(os.path.join(args.data_dir, "ratings_test.json")):
        prepare_train(args, vocab, os.path.join(args.data_dir, "ratings_test.txt"), os.path.join(args.data_dir, "ratings_test.json"))
