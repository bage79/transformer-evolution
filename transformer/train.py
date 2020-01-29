import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import optimization as optim
from config import IS_MAC, Config
from transformer import data
from transformer import model as transformer
from transformer.model import MovieClassification
from vocab import load_vocab


def set_seed(args):
    """ random seed """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_process_group(rank, world_size):
    """ init_process_group """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def destroy_process_group():
    """ destroy_process_group """
    dist.destroy_process_group()


def eval_epoch(config, rank, classification: MovieClassification, data_loader):
    """ 모델 epoch 평가 """
    matchs = []
    classification.eval()

    for i, value in enumerate(data_loader):
        labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

        outputs = classification(enc_inputs, dec_inputs)
        logits = outputs[0]  # logits=(bs, n_outputs)
        _, indices = logits.max(1)  # indices=(bs)

        match = torch.eq(indices, labels).detach()
        matchs.extend(match.cpu())
        accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0
    return accuracy


def train_epoch(args, config, rank, epoch, classification: MovieClassification, criterion, optimizer, scheduler, train_loader):
    """ 모델 epoch 학습 """
    losses = []
    classification.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) epoch={epoch}") as pbar:
        for i, value in enumerate(train_loader):
            # labels=(bs), enc_inputs=(bs, n_enc_seq), dec_inputs=(bs, n_dec_seq)
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)  # (bs), (bs, max_sequence_len), (bs, 1)

            outputs = classification(enc_inputs, dec_inputs)
            logits = outputs[0]  # logits=(bs, n_outputs)

            loss = criterion(logits, labels)  # labels=(bs)
            # if args.gradient_accumulation > 1:
            #     loss /= args.gradient_accumulation
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

            loss.backward()
            if len(losses) % args.gradient_accumulation == 0:
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
    return np.mean(losses)


def train_model(rank, world_size, args):
    """ 모델 학습 """
    if 1 < args.n_gpu:
        init_process_group(rank, world_size)
    master = (world_size == 0 or rank % world_size == 0)
    if master and args.wandb:
        wandb.init(project=args.project, resume=args.name, tags=args.tags)

    vocab = load_vocab(args.vocab)

    config = Config.load(args.config)
    config.n_enc_vocab, config.n_dec_vocab = len(vocab), len(vocab)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

    best_epoch, best_loss, best_score = 0, 0, 0
    model: MovieClassification = transformer.MovieClassification(config)
    if os.path.isfile(args.save):
        best_epoch, best_loss, best_score = model.load(args.save)
        print(f"rank: {rank}, last epoch: {best_epoch} load state dict from: {os.path.basename(args.save)}")
    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model.to(config.device)
    if master and args.wandb:
        wandb.watch(model)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, train_sampler = data.build_data_loader(rank, vocab, os.path.abspath(os.path.join(os.getcwd(), args.data_dir, "ratings_train.json")), args, shuffle=True)
    test_loader, _ = data.build_data_loader(rank, vocab, os.path.abspath(os.path.join(os.getcwd(), args.data_dir, "ratings_test.json")), args, shuffle=False)

    t_total = len(train_loader) * args.epoch  # t_total=총 backward() 회수
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    with tqdm(initial=best_epoch + 1, total=args.epoch, position=0) as pbar:
        for epoch in range(best_epoch + 1, args.epoch + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            loss = train_epoch(args, config, rank, epoch, model, criterion, optimizer, scheduler, train_loader)
            score = eval_epoch(config, rank, model, test_loader)
            if master and args.wandb:
                wandb.config.update(args)
                wandb.log(row={"train loss": loss, "accuracy": score}, step=epoch)

            if master:
                if best_score < score:
                    best_epoch, best_loss, best_score = epoch, loss, score
                    pbar.set_description(f'Best (score={best_score:.3f}, epoch={best_epoch})')
                    if isinstance(model, DistributedDataParallel):
                        model.module.save(best_epoch, best_loss, best_score, args.save)
                    else:
                        model.save(best_epoch, best_loss, best_score, args.save)
                else:
                    if best_epoch + 5 < epoch:  # early stop
                        break

            pbar.update()

    wandb.save(args.name)
    if 1 < args.n_gpu:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data' if not IS_MAC else '../data_sample', type=str, required=False,
                        help='a data directory which have downloaded, corpus text, vocab files.')
    parser.add_argument('--vocab', default='kowiki.model', type=str, required=False,
                        help='vocab file')
    parser.add_argument('--config', default='config_half.json', type=str, required=False,
                        help='config file')
    parser.add_argument('--epoch', default=20 if not IS_MAC else 4, type=int, required=False,
                        help='max epoch')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False,
                        help='real batch size = gradient_accumulation_steps * batch')
    parser.add_argument('--batch', default=256 if not IS_MAC else 4, type=int, required=False,
                        help='batch')  # batch=256 for Titan XP, batch=512 for V100
    parser.add_argument('--gpu', default=None, type=int, required=False,
                        help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='random seed for initialization')
    parser.add_argument('--weight_decay', type=float, default=0, required=False,
                        help='weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False,
                        help='learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False,
                        help='adam epsilon')
    parser.add_argument('--warmup_steps', type=float, default=0, required=False,
                        help='warmup steps')
    parser.add_argument('--wandb', default=False, type=bool, required=False, help='logging to wandb or not')
    parser.add_argument('--project', default='transformer-evolution-bage', type=str, required=False, help='project name for wandb')
    parser.add_argument('--tags', default=['transformer'], type=list, required=False, help='tags for wandb')
    args = parser.parse_args()

    # same wandb run_id cause Error 1062: Duplicate entry
    now = datetime.datetime.now()
    args.name = f'transformer.{os.path.basename(args.data_dir)}.{os.path.basename(args.config).replace(".json", "")}.batch.{args.batch * args.gradient_accumulation}.date.{now.year:4d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'

    args.vocab = os.path.abspath(os.path.join(os.getcwd(), args.data_dir, args.vocab))
    args.save = os.path.abspath(os.path.join(os.getcwd(), "save", f'{args.name}.pth'))
    print(args)

    if not os.path.exists("save"):
        os.makedirs("save")

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0
    set_seed(args)

    if 1 < args.n_gpu:
        mp.spawn(train_model,
                 args=(args.n_gpu, args),
                 nprocs=args.n_gpu,
                 join=True)
    else:
        train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)
