import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import optimization as optim
from albert import data
from albert.model import ALBERTPretrain
from config import Config, get_model_filename, is_mac_or_pycharm, pretty_args
from vocab import load_vocab


def set_seed(args):
    """ random seed """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(args.seed)


def init_process_group(rank, world_size):
    """ init_process_group """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def destroy_process_group():
    """ destroy_process_group """
    dist.destroy_process_group()


def train_epoch(config, rank, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader):
    """ 모델 epoch 학습 """
    losses = []
    model.train()

    for value in tqdm(train_loader, desc=f"pretrain({rank})", position=1, leave=False):
        labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(config.device), value)

        optimizer.zero_grad()
        outputs = model(inputs, segments)
        logits_cls, logits_lm = outputs[0], outputs[1]

        loss_cls = criterion_cls(logits_cls, labels_cls)
        loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
        loss = loss_cls + loss_lm

        loss_val = loss_lm.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(losses)


def train_model(rank, world_size, args):
    """ 모델 학습 """
    if 1 < args.n_gpu:
        init_process_group(rank, world_size)
    master = (world_size == 0 or rank % world_size == 0)
    if master and args.wandb:
        wandb.init(project=args.project)

    vocab = load_vocab(args.vocab)

    config = Config.load(args.config)
    config.n_enc_vocab = len(vocab)
    config.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(config)

    best_epoch, best_loss = 0, 0
    train_model = ALBERTPretrain(config)
    if os.path.isfile(args.pretrain_save):
        try:
            best_epoch, best_loss = train_model.albert.load(args.pretrain_save)
            print(f"load pretrain from: {os.path.basename(args.pretrain_save)}, epoch={best_epoch}, loss={best_loss:.4f}")
        except:
            print(f'load {os.path.basename(args.pretrain_save)} failed.')

    if 1 < args.n_gpu:
        train_model.to(config.device)
        # noinspection PyArgumentList
        train_model = DistributedDataParallel(train_model, device_ids=[rank], find_unused_parameters=True)
    else:
        train_model.to(config.device)

    if master and args.wandb:
        wandb.watch(train_model)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()

    train_loader: DataLoader = data.build_pretrain_loader(vocab, args, shuffle=True)

    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    start_epoch = best_epoch + 1
    losses = []
    with trange(args.epoch, desc="Epoch", position=0) as pbar:
        pbar.set_postfix_str(f"best epoch: {best_epoch}, loss: {best_loss:.4f}")
        for step in pbar:
            epoch = step + start_epoch

            loss = train_epoch(config, rank, train_model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader)
            losses.append(loss)
            if master and args.wandb:
                wandb.log({"loss": loss})

            if master:
                best_epoch, best_loss = epoch, loss
                if isinstance(train_model, DistributedDataParallel):
                    train_model.module.albert.save(best_epoch, best_loss, args.pretrain_save)
                else:
                    train_model.albert.save(best_epoch, best_loss, args.pretrain_save)

                pbar.set_postfix_str(f"best epoch: {best_epoch}, loss: {best_loss:.4f}")

    if 1 < args.n_gpu:
        destroy_process_group()


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), '../data') if not is_mac_or_pycharm() else os.path.join(os.getcwd(), '../data_sample')
    data_dir = os.path.abspath(data_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=data_dir, type=str, required=False,
                        help='a data directory which have downloaded, corpus text, vocab files.')
    parser.add_argument('--vocab', default=os.path.join(data_dir, 'kowiki.model'), type=str, required=False,
                        help='vocab file')
    parser.add_argument('--pretrain', default=os.path.join(data_dir, 'kowiki.albert.json'), type=str, required=False,
                        help='input pretrain data file')
    parser.add_argument('--pretrain_save', default='albert.pth', type=str, required=False,
                        help='save file')

    parser.add_argument('--config', default='config_half.json' if not is_mac_or_pycharm() else 'config_min.json', type=str, required=False,
                        help='config file')
    parser.add_argument('--epoch', default=20 if not is_mac_or_pycharm() else 5, type=int, required=False,
                        help='epoch')
    parser.add_argument('--batch', default=256 if not is_mac_or_pycharm() else 4, type=int, required=False,
                        help='batch')
    parser.add_argument('--gpu', default=None, type=int, required=False,
                        help='GPU id to use.')
    parser.add_argument('--seed', default=42, type=int, required=False,
                        help='random seed for initialization')
    parser.add_argument('--wandb', default=False, type=bool, required=False, help='logging to wandb or not')
    parser.add_argument('--project', default='albert-pretrain', type=str, required=False, help='project name for wandb')
    args = parser.parse_args()
    args.pretrain_save = os.path.abspath(os.path.join(os.getcwd(), 'pretrain_save', f'{get_model_filename(args)}.pth'))
    print(pretty_args(args, sys.argv))

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0
    set_seed(args)

    if not os.path.exists(os.path.dirname(args.pretrain_save)):
        os.makedirs(os.path.dirname(args.pretrain_save))

    if 1 < args.n_gpu:
        # noinspection PyTypeChecker
        mp.spawn(train_model,
                 args=(args.n_gpu, args),
                 nprocs=args.n_gpu,
                 join=True)
    else:
        train_model(rank=0 if args.gpu is None else args.gpu, world_size=args.n_gpu, args=args)
