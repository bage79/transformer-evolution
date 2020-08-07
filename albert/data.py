import argparse
import json
import os
from random import randrange, shuffle, choice, choices, random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import is_mac_or_pycharm, Config
from vocab import load_vocab

SPAN_LEN = 3
SPAN_VALUE = np.array([i + 1 for i in range(SPAN_LEN)])
SPAN_RATIO = np.array([1 / i for i in SPAN_VALUE])
SPAN_RATIO /= np.sum(SPAN_RATIO)


def get_span_length():
    """ SPAN 길이 """
    return choices(SPAN_VALUE, SPAN_RATIO)[0]


def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    """ 마스크 생성 """
    cand_idx = {}
    index = 0
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[index].append(i)
        else:
            index += 1
            cand_idx[index] = [i]
    keys = list(cand_idx.keys())
    shuffle(keys)

    mask_lms = []
    covered_idx = set()
    for index in keys:
        if len(mask_lms) >= mask_cnt:
            break
        span_len = get_span_length()
        if len(cand_idx) <= index + span_len:
            continue
        index_set = []
        for i in range(span_len):
            index_set.extend(cand_idx[index + i])
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        is_idx_covered = False
        for index in index_set:
            if index in covered_idx:
                is_idx_covered = True
                break
        if is_idx_covered:
            continue

        for index in index_set:
            covered_idx.add(index)
            masked_token = None
            if random() < 0.8:  # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random() < 0.5:  # 10% keep original
                    masked_token = tokens[index]
                else:  # 10% random word
                    masked_token = choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


def trim_tokens(tokens_a, tokens_b, max_seq):
    """ 쵀대 길이 초과하는 토큰 자르기 """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break

        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()


def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
    """ pretrain 데이터 생성 """
    # for CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq

    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i])  # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 1 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                if len(current_chunk) == 1 or random() < 0.5:
                    is_next = 0
                    # switch sentence order
                    tokens_tmp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = tokens_tmp
                else:
                    is_next = 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


def make_pretrain_data(args):
    """ pretrain 데이터 생성 """
    if os.path.isfile(args.pretrain):
        print(f"{args.pretrain} exists")
        return

    vocab = load_vocab(args.vocab)
    vocab_list = [vocab.id_to_piece(wid) for wid in range(vocab.get_piece_size()) if not vocab.is_unknown(wid)]

    docs = []
    with open(args.corpus, "r") as f:
        lines = f.read().splitlines()

        doc = []
        for line in tqdm(lines, desc=f"Loading {os.path.basename(args.corpus)}"):
            line = line.strip()
            if line == "":
                if 0 < len(doc):
                    docs.append(doc)
                    doc = []
            else:
                pieces = vocab.encode_as_pieces(line)
                if 0 < len(pieces):
                    doc.append(pieces)
        if 0 < len(doc):
            docs.append(doc)

    with open(args.pretrain, "w") as out_f:
        config = Config.load(args.config)
        for i, doc in enumerate(tqdm(docs, desc=f"Making {os.path.basename(args.pretrain)}", unit=" lines")):
            instances = create_pretrain_instances(docs, i, doc, config.n_enc_seq, config.mask_prob, vocab_list)
            for instance in instances:
                out_f.write(json.dumps(instance, ensure_ascii=False))
                out_f.write("\n")


class PretrainDataSet(torch.utils.data.Dataset):
    """ pretrain 데이터셋 """

    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []

        with open(infile, "r") as f:
            for line in tqdm(f.read().splitlines(), desc=f"Loading {os.path.basename(infile)}", unit=" lines", position=2, leave=False):
                instance = json.loads(line)
                self.labels_cls.append(instance["is_next"])
                sentences = [vocab.piece_to_id(p) for p in instance["tokens"]]
                self.sentences.append(sentences)
                self.segments.append(instance["segment"])
                mask_idx = np.array(instance["mask_idx"], dtype=np.int)
                mask_label = np.array([vocab.piece_to_id(p) for p in instance["mask_label"]], dtype=np.int)
                label_lm = np.full(len(sentences), dtype=np.int, fill_value=-1)
                label_lm[mask_idx] = mask_label
                self.labels_lm.append(label_lm)

    def __len__(self):
        assert len(self.labels_cls) == len(self.labels_lm) == len(self.sentences) == len(self.segments)
        return len(self.labels_cls)

    def __getitem__(self, item):
        return (torch.tensor(self.labels_cls[item]),
                torch.tensor(self.labels_lm[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


def pretrin_collate_fn(inputs):
    """ pretrain data collate_fn """
    labels_cls, labels_lm, inputs, segments = list(zip(*inputs))

    labels_cls = torch.stack(labels_cls, dim=0)
    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        labels_cls,
        labels_lm,
        inputs,
        segments
    ]
    return batch


def build_pretrain_loader(vocab, args, shuffle=True) -> DataLoader:
    """ pretraun 데이터 로더 """
    dataset = PretrainDataSet(vocab, args.pretrain)
    if 1 < args.n_gpu and shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, collate_fn=pretrin_collate_fn)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=shuffle, collate_fn=pretrin_collate_fn)
    return loader


class MovieDataSet(torch.utils.data.Dataset):
    """ 영화 분류 데이터셋 """

    def __init__(self, vocab, infile, data_type):
        self.vocab = vocab
        self.labels = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"load {data_type}", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                sentence = [vocab.piece_to_id("[CLS]")] + [vocab.piece_to_id(p) for p in data["doc"]] + [vocab.piece_to_id("[SEP]")]
                self.sentences.append(sentence)
                self.segments.append([0] * len(sentence))

    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        assert len(self.labels) == len(self.segments)
        return len(self.labels)

    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


def movie_collate_fn(inputs):
    """ movie data collate_fn """
    labels, inputs, segments = list(zip(*inputs))

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        inputs,
        segments,
    ]
    return batch


def build_data_loader(vocab, infile, args, data_type='train', shuffle=True) -> DataLoader:
    """ 데이터 로더 """
    dataset = MovieDataSet(vocab, infile, data_type)
    if 1 < args.n_gpu and shuffle:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, collate_fn=movie_collate_fn)
    else:
        sampler = None
        # noinspection PyTypeChecker
        loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=shuffle, collate_fn=movie_collate_fn)
    return loader


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), '../data') if not is_mac_or_pycharm() else os.path.join(os.getcwd(), '../data_sample')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=data_dir, type=str, required=False,
                        help='a data directory which have downloaded, corpus text, vocab files.')
    parser.add_argument("--corpus", default=os.path.join(data_dir, "kowiki.txt"), type=str, required=False,
                        help="input text file")
    parser.add_argument("--pretrain", default=os.path.join(data_dir, "kowiki.albert.json"), type=str, required=False,
                        help="output json file")
    parser.add_argument("--vocab", default=os.path.join(data_dir, "kowiki.model"), type=str, required=False,
                        help="vocab file")

    parser.add_argument('--config', default='config_half.json' if not is_mac_or_pycharm() else 'config_min.json', type=str, required=False,
                        help='config file')
    args = parser.parse_args()

    make_pretrain_data(args)
