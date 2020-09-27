import pickle
import argparse

import math
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
import cy_heuristics as heu
import sched_heuristic as py_heu
from sklearn.utils import shuffle
import scipy.stats
from sched_heuristic import scores_to_priority

parser = argparse.ArgumentParser()
parser.add_argument("--num_tasks", type=int)
parser.add_argument("--num_procs", type=int)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--num_train_dataset", type=int, default=300000)
parser.add_argument("--num_test_dataset", type=int, default=3000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1.0 * 1e-4)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", type=int, default=0)
parser.add_argument("--range_l", type=str, default="2.10")
parser.add_argument("--range_r", type=str, default="2.50")


confidence = 0.05

args = parser.parse_args()

use_deadline = args.use_deadline == 1
test_module = heu.test_Lee
use_cuda=True


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
    if use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False

    util_range = get_util_range(args.num_procs)
    print(util_range)
    trsets = []
    tesets = []
    on = False
    for util in util_range:
        print(util)
        if util == args.range_l:
            on = True
        if on:
            with open("np_tr/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
                print(util)

            with open("np_te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                tesets.append(ts)
        if util == args.range_r:
            break

    print(len(trsets))
    train_dataset = Datasets(trsets)
    test_dataset = Datasets(tesets)

    train_dataset.setlen(args.num_train_dataset)
    test_dataset.setlen(args.num_test_dataset)
    print(len(train_dataset))
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    print(len(train_data_loader))
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculating heuristics
    temp_fname = "%s-%s" % (args.num_procs, args.num_tasks)
    require_pt = False
    try:
        model = torch.load("models/" + temp_fname).cuda()

        model.train()
    except:
        require_pt = True
        print("No previous model!")
        model = Solver(args.num_procs,
                   args.embedding_size,
                   args.hidden_size,
                   args.num_tasks,
                   use_deadline=use_deadline,
                   use_cuda=use_cuda)
    bl_model = Solver(args.num_procs,
                   args.embedding_size,
                   args.hidden_size,
                   args.num_tasks,
                   use_deadline=use_deadline,
                   use_cuda=use_cuda)
    bl_model.load_state_dict(model.state_dict())
    if use_cuda:
        model = model.cuda()
        bl_model = bl_model.cuda()

    bl_model = bl_model.eval()

    def wrap(x):
        _sample, num_proc, use_deadline = x
        order = py_heu.get_DM_scores(_sample, num_proc, use_deadline=use_deadline)
        priority = scores_to_priority(order)
        return test_module(_sample, num_proc, priority, use_deadline, 0)


    with ProcessPoolExecutor(max_workers=4) as executor:
        inputs = []
        res_opa = np.zeros(len(test_dataset), dtype=int).tolist()
        for i, sample in test_dataset:
            #ret = heu.OPA(sample, args.num_procs, heu.test_DA_LC, use_deadline)
            inputs.append((sample, args.num_procs, use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)

    print("[before training][DM  with Lee14 generates %d]" % opares)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_ = 0
    avg_hit = []
    updates = 0
    prev_ = -1
    if require_pt:
        for batch_idx, (_, sample_batch) in enumerate(train_data_loader):
            guide = []
            as_np = sample_batch.numpy()
            for r in as_np:
                guide.append(np.argsort(py_heu.get_DM_scores(r, args.num_procs, use_deadline)))
            guide = torch.from_numpy(np.array(guide, dtype=np.int64))
            if use_cuda:
                guide = guide.cuda()
                sample_batch = sample_batch.cuda()
            num_samples = sample_batch.shape[0]
            optimizer.zero_grad()
            rewards, log_probs, action = model.forward_np(sample_batch, guide=guide)
            loss = -torch.sum((rewards * log_probs), dim=-1).mean()
            loss.backward()
            loss_ += loss.cpu().detach().numpy()
            avg_hit.append((rewards.cpu().detach().mean()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            updates += 1
            if (updates % 50 == 0):
                model.eval()
                ret = []
                for i, _batch in eval_loader:
                    if use_cuda:
                        _batch = _batch.cuda()
                    R, log_prob, actions = model.forward_np(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for i in range(args.num_tasks):
                            order[chosen[i]] = args.num_tasks - i - 1
                        if use_cuda:
                            ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
                        else:
                            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
                fname = "p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
                rl_model_sum = np.sum(ret)
                print("[consumed %d samples][PRETRAINING][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
                if rl_model_sum <= prev_ or updates >= 50:
                    print("RL pretraining end")
                    break
                prev_ = rl_model_sum
                model.train()
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
            if use_cuda:
                sample_batch = sample_batch.cuda()
            num_samples = sample_batch.shape[0]
            optimizer.zero_grad()
            rewards, log_probs, action = model.forward_np(sample_batch)
            baseline, _bl_log_probs, _bl_action = bl_model.forward_np(sample_batch, argmax=True)
            advantage = rewards - baseline
            #log_probs = torch.sum(log_probs, dim=-1)
            #log_probs[log_probs < -100] = -100
            loss = -torch.sum((advantage * log_probs), dim=-1).mean()
            loss.backward()
            loss_ += loss.cpu().detach().numpy()
            avg_hit.append((rewards.cpu().detach().mean()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            updates += 1
            ## baseline update
            #diff = (log_probs - _bl_log_probs.sum(dim=-1))
            if use_cuda:
                diff = advantage.sum(dim=-1).detach().cpu().numpy()
            else:
                diff = advantage.sum(dim=-1).detach().numpy()
            D = diff.mean()
            S_D = 1e-10 + np.sqrt(((diff - D) ** 2).sum() / (1e-10 + num_samples - 1))
            tval = D / (S_D / (1e-10 + math.sqrt(1e-10 + num_samples)))
            p = scipy.stats.t.cdf(tval, num_samples)
            if (p >= 1. - 0.5 * confidence) or (p <= 0.5 * confidence):
                bl_model.load_state_dict(model.state_dict())
            if (updates % 100 == 0):
                model.eval()
                ret = []
                for i, _batch in eval_loader:
                    if use_cuda:
                        _batch = _batch.cuda()
                    R, log_prob, actions = model.forward_np(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for i in range(args.num_tasks):
                            order[chosen[i]] = args.num_tasks - i - 1
                        if use_cuda:
                            ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
                        else:
                            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
                fname = "p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
                rl_model_sum = np.sum(ret)

                print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
                stop = False
                with open("res_temp_storage/" + fname, 'a') as f:
                    print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit), file=f)
                    if (rl_model_sum == args.num_test_dataset):
                        print("total hit at epoch", epoch, file=f)
                        print("total hit at epoch", epoch)
                        torch.save(model, "models/" + fname + ".torchmodel")
                        #torch.save(model, "models/" + temp_fname)
                        stop = True

                    if rl_model_sum > _max:
                        noupdateinarow = 0
                        _max = rl_model_sum
                        torch.save(model, "models/" +  fname + ".torchmodel")
                        #torch.save(model, "models/" + temp_fname)
                    else:
                        noupdateinarow += 1
                    if noupdateinarow >= 20:
                        print("not update 20 times", epoch, file=f)
                        print("not update 20 times", epoch)
                        torch.save(model, "models/" +  fname + ".torchmodel")
                        #torch.save(model, "models/" + temp_fname)
                        stop = True
                if stop:
                    raise NotImplementedError

                model.train()

    model.eval()
    ret = []
    for i, _batch in eval_loader:
        R, log_prob, actions = model.forward_np(_batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for i in range(args.num_tasks):
                order[chosen[i]] = args.num_tasks - i - 1
            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
    fname = "p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    rl_model_sum = np.sum(ret)

    print("[at epoch %d][RL model generates %d][OPA generates %d]" % (epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
    with open("res_temp_storage/" + fname, 'a') as f:
        print("[at epoch %d][RL model generates %d][OPA generates %d]" % (epoch, rl_model_sum, opares), "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit), file=f)

    if rl_model_sum > last_rl_model_sum:
        torch.save(model,"models/" +  fname + ".torchmodel")
        #torch.save(model, "models/" + temp_fname)
        last_rl_model_sum = rl_model_sum
    model.train()
