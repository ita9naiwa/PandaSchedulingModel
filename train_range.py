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
from concurrent.futures import ProcessPoolExecutor
from sched_solver import Solver
from sched import SchedT1Dataset
#import sched_heuristic as heu
#import sched_heuristic
import cy_heuristics as heu
from sklearn.utils import shuffle
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int)
parser.add_argument("--num_procs", type=int)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--num_train_dataset", type=int, default=1000)
parser.add_argument("--num_test_dataset", type=int, default=500)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--lr", type=float, default=1.0 * 1e-4)
parser.add_argument("--baseline_iter", type=int, default=10)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", type=bool, default=False)
parser.add_argument("--range_l", type=str, default="0.10")
parser.add_argument("--range_r", type=str, default="0.60")
parser.add_argument("--model", type=int, defaul=0)

args = parser.parse_args()
use_deadline = args.use_deadline
print(bool(args.model))
test_module = heu.test_RTA_LC

def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])
    return ret

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

    util_range = get_util_range(args.num_procs)
    trsets = []
    tesets = []
    for util in util_range:
        on = False
        if util == args.range_l:
            on = True
        if on:
            with open("tr/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
            with open("te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                tesets.append(ts)
        if util == args.range_r:
            break

    train_dataset = Datasets(trsets)
    test_dataset = Datasets(tesets)

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
                   use_deadline=use_deadline
                   nimp=bool(args.model))

    if args.use_cuda:
        model = model.cuda()

    res_opa = []
    for i, sample in tqdm(test_dataset):
        ret = heu.OPA(sample, args.num_procs, heu.test_DA_LC, use_deadline)
        res_opa.append(ret)
    opares = np.sum(res_opa)
    print("[before training][OPA generates %d]" % opares)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=0.9, last_epoch=-1)
    last_rl_model_sum = -1
    updates = 0
    noupdateinarow = 0
    _max = -1
    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = []
        for batch_idx, (_, sample_batch) in enumerate(train_data_loader):
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
            scheduler.step()
            updates += 1
            if (updates % 50 == 0):
                model.eval()
                ret = []
                for i, _batch in eval_loader:
                    R, log_prob, actions = model(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for i in range(args.num_tasks):
                            order[chosen[i]] = args.num_tasks - i - 1
                        ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
                fname = "p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
                rl_model_sum = np.sum(ret)

                print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, epoch, rl_model_sum, opares), "log_probability\t",log_prob.cpu().detach().numpy().mean(), "loss", loss_, "avg_hit", np.mean(avg_hit))
                stop = False
                with open("res_temp_storage/" + fname, 'a') as f:
                    print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, epoch, rl_model_sum, opares), "log_probability\t",log_prob.cpu().detach().numpy().mean(), "loss", loss_, "avg_hit", np.mean(avg_hit), file=f)
                    if (rl_model_sum == args.num_test_dataset):
                        print("total hit at epoch", epoch, file=f)
                        print("total hit at epoch", epoch)
                        torch.save(model, "models/" +  fname + ".torchmodel")
                        stop = True

                    if rl_model_sum > _max:
                        noupdateinarow = 0
                        _max = rl_model_sum
                        torch.save(model, "models/" +  fname + ".torchmodel")
                    else:
                        noupdateinarow += 1
                    if noupdateinarow >= 10:
                        print("not update 10 times", epoch, file=f)
                        print("not update 10 times", epoch)
                        torch.save(model, "models/" +  fname + ".torchmodel")
                        stop = True
                if stop:
                    raise NotImplementedError

                model.train()

    model.eval()
    ret = []
    for i, _batch in eval_loader:
        R, log_prob, actions = model(_batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for i in range(args.num_tasks):
                order[chosen[i]] = args.num_tasks - i - 1
            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
    fname = "p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    rl_model_sum = np.sum(ret)

    print("[at epoch %d][RL model generates %d][OPA generates %d]" % (epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "loss", loss_, "avg_hit", np.mean(avg_hit))
    with open("res_temp_storage/" + fname, 'a') as f:
        print("[at epoch %d][RL model generates %d][OPA generates %d]" % (epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "loss", loss_, "avg_hit", np.mean(avg_hit), file=f)

    if rl_model_sum > last_rl_model_sum:
        torch.save(model,"models/" +  fname + ".torchmodel")
        last_rl_model_sum = rl_model_sum
    model.train()