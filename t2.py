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
import sched_heuristic as heu
from sklearn.utils import shuffle
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--num_train_dataset", type=int, default=500)
parser.add_argument("--num_test_dataset", type=int, default=100)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--lr", type=float, default=1.0 * 1e-4)
parser.add_argument("--baseline_iter", type=int, default=10)
args = parser.parse_args()
use_deadline = False

test_module = heu.test_RTA_LC
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

    with open("tr/%d-%d/6.20" % (args.num_procs, args.num_tasks), 'rb') as f:
        train_dataset = pickle.load(f)


    with open("te/%d-%d/6.20" % (args.num_procs, args.num_tasks), 'rb') as f:
        test_dataset = pickle.load(f)
    train_dataset.setlen(args.num_train_dataset)
    test_dataset.setlen(args.num_test_dataset)

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

    res_opa = []
    for i, sample in tqdm(test_dataset):
        ret, _ = heu.OPA(sample, args.num_procs, heu.test_DA_LC, use_deadline)
        res_opa.append(ret)

    print("[before training][OPA generates %d]" % np.sum(res_opa))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = []
        for batch_idx, (_, sample_batch) in enumerate(train_data_loader):
            print(batch_idx)
            if args.use_cuda:
                sample_batch.cuda()
            rtt = []
            optimizer.zero_grad()
            for t in range(args.baseline_iter):
                rewards, log_probs, action = model(sample_batch)
                rtt.append((rewards, log_probs, action))
            rs = [x[0] for x in rtt]
            baseline = torch.mean(torch.stack(rs, dim=0), dim=0)

            for rewards, log_probs, action in rtt:
                advantage = rewards - baseline
                log_probs = torch.sum(log_probs, dim=-1)
                log_probs[log_probs < -100] = -100
                loss = -(advantage * log_probs).mean()
                loss /= args.baseline_iter
                loss.backward()
                loss_ += loss.cpu().detach().numpy()
                avg_hit.append((rewards).detach().mean())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        print("loss", loss_, "avg_hit", np.mean(avg_hit))

        model.eval()
        ret = []
        for i, _batch in eval_loader:
            R, log_prob, actions = model(_batch, argmax=True)
            for j, chosen in enumerate(actions.cpu().numpy()):
                order = np.zeros_like(chosen)
                for i in range(args.num_tasks):
                    order[chosen[i]] = args.num_tasks - i - 1
                ret.append(test_module(_batch[j], args.num_procs, order, use_deadline, False))
        print("log_probability\t", log_prob.cpu().detach().numpy().mean())
        print("[at epoch %d][RL model generates %d][OPA generates %d]" % (epoch, np.sum(ret), np.sum(res_opa)))
        torch.save(model, "p4-t20")
        model.train()
