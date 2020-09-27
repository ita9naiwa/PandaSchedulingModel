import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sched_solver import Solver
from sched import SchedSingleDataset
from sched_heuristic import get_rm_solution
from sched_heuristic import liu_test

parser = argparse.ArgumentParser()

parser.add_argument("--seq_len", type=int, default=24)
parser.add_argument("--num_procs", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_tr_dataset", type=int, default=2000)
parser.add_argument("--num_te_dataset", type=int, default=1000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--beta", type=float, default=0.9)


args = parser.parse_args()


def run(train_dataset, test_dataset, model):
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)
    eval_loader = DataLoader(test_dataset, batch_size=args.num_te_dataset, shuffle=False)

    # Calculating heuristics
    liu_boundary = torch.zeros(args.num_te_dataset)
    for i, pointset in tqdm(test_dataset):
        liu_boundary[i] = float(liu_test(pointset, num_procs=args.num_procs))
    heuristic_distance = torch.zeros(args.num_te_dataset)
    for i, pointset in tqdm(test_dataset):
        heuristic_distance[i] = get_rm_solution(pointset, num_procs=args.num_procs)



    if args.use_cuda:
        model = model.cuda()


    # Train loop
    moving_avg = torch.zeros(args.num_tr_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()
    #generating first baseline
    cc = 1
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        for _ in range(cc):
            rewards, _, _ = model(sample_batch)
            moving_avg[indices] += rewards.float()
    moving_avg /= cc

    model.eval()
    ret = []

    for i, batch in eval_loader:
        if args.use_cuda:
            batch = batch.cuda()
        R, _, _ = model(batch, argmax=True)
    print("[before training][RL model generates %d][heuristic generates %d][liu generates %d]" %(
        (R > 0).sum().detach().numpy(),
        (heuristic_distance > 0).sum().numpy(),
        (liu_boundary > 0).sum().numpy()))


    #Training
    model.train()

    # Guided Learning, First Step
    optimizer = optim.Adam(model.parameters(), lr=5.0 * 1e-5)
    model.train()
    for epoch in range(args.num_epochs):
        loss_ = 0
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = -(advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_ += loss.detach().numpy()
        print("loss", loss_)




        model.eval()
        ret = []
        for i, batch in eval_loader:
            if args.use_cuda:
                batch = batch.cuda()
            R, log_prob, _ = model(batch, argmax=True)
        print("log_probability\t", log_prob.detach().numpy().mean())
        print("[at epoch %d][RL model generates %d][heuristic generates %d][liu generates %d]" %(
            epoch,
            (R > 0).sum().detach().numpy(),
            (heuristic_distance > 0).sum().numpy(),
            (liu_boundary > 0).sum().numpy()))
        #print("AVG R", R.float().mean().detach().numpy())
        model.train()
        #if (R > 0).sum() >= len(test_dataset) // 2:
        #    break

if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False
    model = Solver(args.num_procs,
                   2,
                   args.embedding_size,
                   args.hidden_size,
                   args.seq_len)

    for level in [0.5, 0.7, 0.8, 0.9, 0.95]:
        train_dataset = SchedSingleDataset(args.num_procs, args.seq_len, args.num_tr_dataset, util_range=(0.6, 0.95))
        test_dataset = SchedSingleDataset(args.num_procs, args.seq_len, args.num_te_dataset, util_range=(0.6, 0.95))
        run(train_dataset, test_dataset, model)
