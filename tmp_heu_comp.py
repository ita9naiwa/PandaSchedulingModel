import time
import argparse
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sched_heuristic as oldh
import cy_heuristics as newh

import pickle
from scipy.stats import kendalltau as tau, spearmanr as rho
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--num_proc", type=int, default=4)
parser.add_argument("--use_deadline", type=bool, default=True)

args = parser.parse_args()

num_proc = args.num_proc
use_deadline = args.use_deadline

print("num_proc=", num_proc)
print("use_deadline", use_deadline)

def scores_to_priority(scores):
    rank = np.argsort(-scores)
    priority = np.zeros_like(rank)
    for i in range(len(priority)):
        priority[rank[i]] = i
    return priority

def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])

    return ret


num_tasks_list = [16]

res_map = defaultdict(lambda: defaultdict(lambda: 0))
tmap = defaultdict(lambda: 0)
for num_tasks in num_tasks_list:
    if num_proc >= num_tasks:
        continue
    util_range = get_util_range(num_proc)
    for util in ["1.50", "2.50"]:
        print(num_tasks, util)
        res = res_map[(num_proc, num_tasks, util)]
        with open("eval/%d-%d/%s" % (num_proc, num_tasks, util), 'rb') as f:
            train_dataset = pickle.load(f)
        i = 0
        for x, y in train_dataset:

            if i == 5000:
                break
            p =time.time()
            old_opa_res, _ = oldh.OPA(y, num_proc, oldh.test_DA_LC, use_deadline)
            n = time.time() - p
            tmap['old_opa'] += n
            p =time.time()
            new_opa_res = newh.OPA_DA_LC(y, True, num_proc, num_tasks)
            n = time.time() - p
            tmap['new_opa'] += n

            res['old_opa'] += old_opa_res
            res['new_opa'] += new_opa_res

            q = scores_to_priority(oldh.get_DM_DS_scores(y, num_proc, use_deadline=use_deadline))
            p = time.time()
            res['old_da_lc'] += (oldh.test_RTA_LC(y, num_proc, q, use_deadline))
            n = time.time() - p
            tmap['old_da_lc'] += n
            p =time.time()
            res['new_da_lc'] += (newh.test_RTA_LC(y, num_proc, q, use_deadline))
            n = time.time() - p
            tmap['new_da_lc'] += n

            i += 1
        print(res)
        print(tmap)
