import json

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class MovieDataSet(torch.utils.data.Dataset):
    """ 영화 분류 데이터셋 """

    def __init__(self, vocab, infile, rank=0):
        self.vocab = vocab
        self.labels = []
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading({rank}) {infile}", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])

    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)

    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor([self.vocab.piece_to_id("[BOS]")]))


def movie_collate_fn(inputs):
    """ movie data collate_fn """
    # inputs = [(label, [wids in sentence], [BOS]), ...] -> [label, ...], [[wids in sentence], ...], [[BOS], ...]
    labels, _enc_inputs, _dec_inputs = list(zip(*inputs))

    enc_inputs = torch.nn.utils.rnn.pad_sequence(_enc_inputs, batch_first=True, padding_value=0)  # [[wids in sentence], ...] -> [[padded wids in sentence], ...]
    dec_inputs = torch.nn.utils.rnn.pad_sequence(_dec_inputs, batch_first=True, padding_value=0)  # [[BOS], ...] -> [[BOS], ...]

    batch = [
        torch.stack(labels, dim=0),  # [tensor(label), ...] -> tensor([label, ...]) (bs)
        enc_inputs,  # tensor([[padded wids in sentence], ...]) (bs, max_sequence_len)
        dec_inputs,  # tensor([[BOS], ...]) (bs, 1)
    ]
    return batch


def build_data_loader(rank, vocab, infile, args, shuffle=True):
    """ 데이터 로더 """
    dataset = MovieDataSet(vocab, infile, rank=rank)
    sampler = None
    if 1 < args.n_gpu and shuffle:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=args.batch, collate_fn=movie_collate_fn, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=args.batch, collate_fn=movie_collate_fn, shuffle=shuffle)
    return loader, sampler
