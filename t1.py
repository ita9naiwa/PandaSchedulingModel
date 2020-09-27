import pickle
import argparse

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sched_solver import Solver
from sched import SchedT1Dataset
from sched_heuristic import get_rm_solution
from sched_heuristic import liu_test
import sched_heuristic as heu
from sched_heuristic import scores_to_priority
from sklearn.utils import shuffle
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=8)
parser.add_argument("--num_procs", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--num_train_dataset", type=int, default=1000)
parser.add_argument("--num_test_dataset", type=int, default=100)
parser.add_argument("--embedding_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--beta", type=float, default=0.7)
parser.add_argument("--lr", type=float, default=1.0 * 1e-4)
args = parser.parse_args()
use_deadline = True

class Datasets(Dataset):
    def __init__(self, l):
        super(Datasets, self).__init__()
        ret = []
        le = []
        for dd in l:
            ret.append(dd.data_set)
        self.data_set = np.vstack(ret)

    def setlen(self, newlen):
        self.data_set = shuffle(self.data_set)
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]

if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False
    with open("tr/%d-%d/0.80" % (args.num_procs, args.num_tasks), 'rb') as f:
        t1 = pickle.load(f)
    with open("tr/%d-%d/1.00" % (args.num_procs, args.num_tasks), 'rb') as f:
        train_dataset = t2 = pickle.load(f)
    with open("tr/%d-%d/1.20" % (args.num_procs, args.num_tasks), 'rb') as f:
        t3 = pickle.load(f)

    with open("te/%d-%d/1.00" % (args.num_procs, args.num_tasks), 'rb') as f:
        test_dataset = pickle.load(f)

    #train_dataset = Datasets([t1, t2, t3])
    train_dataset.setlen(args.num_train_dataset)
    test_dataset.setlen(args.num_test_dataset)
    train_dataset = test_dataset

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

    eval_loader = DataLoader(test_dataset, batch_size=args.num_test_dataset, shuffle=False)

    # Calculating heuristics

    model = Solver(args.num_procs,
                   args.embedding_size,
                   args.hidden_size,
                   args.num_tasks,
                    use_deadline=use_deadline)

    if args.use_cuda:
        model = model.cuda()


    # Train loop
    moving_avg = torch.zeros(args.num_train_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()
    #generating first baseline
    cc = 1
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        print(rewards)
        moving_avg[indices] = rewards.float()
    model.eval()
    ret = []
    res_tkc = []
    res_rm = []
    res_opa = []
    for i, sample in tqdm(test_dataset):

        scores = heu.get_DkC_scores(sample, args.num_procs)
        priority = scores_to_priority(scores)
        res_tkc.append(heu.test_DA(sample, args.num_procs, priority, use_deadline))

        scores = heu.get_DM_scores(sample, args.num_procs)
        priority = scores_to_priority(scores)
        res_rm.append(heu.test_DA(sample, args.num_procs, priority, use_deadline))

        ret, _ = heu.OPA(sample, args.num_procs, heu.test_DA, use_deadline)
        res_opa.append(ret)

    for i, batch in eval_loader:
        if args.use_cuda:
            batch = batch.cuda()
        R, _, _ = model(batch, argmax=True)

    print("[before training][RL model generates %d][rm generates %d][DkC generates %d][OPA generates %d]" %(
        (R > 1.0).sum().detach().numpy(),
        np.sum(res_rm),
        np.sum(res_tkc),
        np.sum(res_opa)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = 0
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            #print(action[0])
            advantage = rewards - moving_avg[indices]
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)

            #advantage = rewards


            avg_hit += (rewards).mean()
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            #print(log_probs[:5])
            loss = -(advantage * log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_ += loss.detach().numpy()
        print("loss", loss_, "avg_hit", avg_hit.detach().numpy())

        model.eval()
        ret = []
        for i, batch in eval_loader:
            if args.use_cuda:
                batch = batch.cuda()
            R, log_prob, actions = model(batch, argmax=True)
            for j, chosen in enumerate(actions.numpy()):
                order = np.zeros_like(chosen)
                for i in range(args.num_tasks):
                    order[chosen[i]] = args.num_tasks - i
                ret.append(heu.test_DA(batch[j], args.num_procs, order, use_deadline, False))
        print("log_probability\t", log_prob.detach().numpy().mean())
        print("[at epoch %d][RL model generates %d][rm generates %d][DkC generates %d][OPA generates %d]" % (
            epoch,
            np.sum(ret),
            np.sum(res_rm),
            np.sum(res_tkc),
            np.sum(res_opa)))
        torch.save(model, "blah2")
        model.train()
