import numpy as np
from itertools import combinations
from collections import Counter
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from copy import deepcopy
import time


def neighboursize(G, vars):
    n_size = np.sum(G[:, vars], axis=0)
    return n_size


def neighbor(G, var):
    temp_G = (G[:, var] + G.T[:, var])
    n = np.where(temp_G > 0)[0].tolist()
    n = list(set(n) - set([var]))
    return n


def mynchoosek(v, k=None):
    if k == None or v == None or k == 0 or len(v) < k:
        out = []
    else:
        out = list(combinations(v, k))
        out = [list(o) for o in out]
    return out


def graph_dline(G, var_x, var_y):
    for x in var_x:
        for y in var_y:
            G[x, y] = 0
    return G


def subv2ind(size, sub):
    # if isinstance(size, np.int64) or isinstance(size, float) or isinstance(size, np.int) or isinstance(size,
    #                                                                                                    np.int32) or isinstance(
    #     size, np.int8) or isinstance(size,np.intc):
    #     k = np.array([1])
    if not isinstance(size, np.ndarray):
        k = np.array([1])
    else:
        temp1 = np.array([1])
        temp2 = np.cumprod(size[:-1])
        k = np.hstack((temp1, temp2))
        k = k.astype(np.int)
    ndx = np.matmul(sub, k.T) - np.sum(k) + 1
    return ndx


# def rem():
#
# def ind2subv(size, sub):
#     temp1 = np.array([1])
#     temp2 = np.cumprod(size[:-1])
#     k = np.hstack((temp1, temp2))
#     k = k.astype(np.int)
#     for i in range(len(size) - 1, -1, -1):


def count(data, nstates):
    if len(data) == 0 or data.size == 0:
        cidx = []
        return cidx
    cidx = np.zeros(np.prod(nstates), )

    # for n in range(data.shape[1]):
    #     ind = subv2ind(nstates, data[:, n].T) - 1
    #     cidx[ind] = cidx[ind] + 1

    data = data - 1
    t = deepcopy(data[0])
    for i in range(1, len(data)):
        x = np.prod(nstates[:i])
        t += deepcopy(data[i]) * x
    idx, count = np.unique(t, return_counts=True)
    for i in range(len(idx)):
        cidx[idx[i]] = count[i]
    return cidx



def Sort(names):
    new_names = []
    dead_list = []
    for name in names:
        temp = name.split("-", 2)[:-1]
        while len(temp[0]) < 2:
            temp[0] = '0' + temp[0]
        dead_list.append(temp[0])
        new_names.append(temp[0] + temp[1])

    rank = [index for index, value in sorted(list(enumerate(new_names)), key=lambda x: x[1])]
    dead_list = list(np.array(dead_list)[rank])
    for i in range(len(dead_list) - 1):
        if dead_list[i] == dead_list[i + 1]:
            dead_list[i] = True
        else:
            dead_list[i] = False
    dead_list[-1] = False
    return rank, dead_list
