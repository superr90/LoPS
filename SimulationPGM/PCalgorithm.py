import numpy as np
from condindepEmp import *
import copy


def parallize_(data, x, data_num, nstates, opts, G, S, y, z):
    if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                    data[z, :].reshape(-1, data_num), nstates[x],
                    nstates[y], nstates[z], 0, opts)[0]:
        G[x, y] = G[y, x] = 0
        S[x][y] = list(set(S[x][y]) | set(z))
        S[y][x] = S[x][y]


def parallize_skleteton(data, nb, i, x, data_num, opts, nstates, G, S, y):
    subsets = mynchoosek(list(set(nb) - set([y])), i)
    if len(subsets) == 0:
        if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num), [],
                        [nstates[x]],
                        [nstates[y]], [], 0, opts)[0]:
            data[x, y] = G[y, x] = 0
    else:
        # with Pool(processes=4) as pool:
        #     pool.map(partial(parallize_, data, x, data_num, nstates, opts, G, S,y), subsets)
        for j in range(len(subsets)):
            z = subsets[j]
            if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                            data[z, :].reshape(-1, data_num), nstates[x],
                            nstates[y], nstates[z], 0, opts)[0]:
                G[x, y] = G[y, x] = 0
                S[x][y] = list(set(S[x][y]) | set(z))
                S[y][x] = S[x][y]


def PCskletetonData_parallize(data):
    var_num = data.shape[0]
    data_num = data.shape[1]
    nstates = np.max(data, 1).T
    nstates = nstates.astype(np.int)
    opts = {
        "Uxgz": 0.1, "Uygz": 0.1, "Uz": 0.1, "Uxyz": 0.1,
    }

    G = np.ones((var_num, var_num)) - np.diag([1] * var_num)
    var = list(range(0, var_num))
    S = [[] for i in range(var_num)]
    for i in range(var_num):
        for j in range(var_num):
            S[i].append([])

    for i in range(len(var)):
        if (neighboursize(G, var) < i).all():
            break
        for x in var:
            nb = neighbor(G, x)
            for y in nb:
                subsets = mynchoosek(list(set(nb) - set([y])), i)
                if len(subsets) == 0:
                    if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num), [],
                                    [nstates[x]],
                                    [nstates[y]], [], 0, opts)[0]:
                        G[x, y] = G[y, x] = 0
                else:
                    # with Pool(processes=4) as pool:
                    #     pool.map(partial(parallize_, data, x, data_num, nstates, opts, G, S, y), subsets)
                    for j in range(len(subsets)):
                        z = subsets[j]
                        if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                                        data[z, :].reshape(-1, data_num), nstates[x],
                                        nstates[y], nstates[z], 0, opts)[0]:
                            G[x, y] = G[y, x] = 0
                            S[x][y] = list(set(S[x][y]) | set(z))
                            S[y][x] = S[x][y]

        # with Pool(processes=4) as pool:
        #     pool.map(partial(parallize_skleteton, data, nb, i, x, data_num, opts, nstates, G, S), nb)
    return G, S


def PCskletetonData(data):
    var_num = data.shape[0]
    data_num = data.shape[1]
    nstates = np.max(data, 1).T
    nstates = nstates.astype(np.int64)
    init = max(0.0002 * data_num, 1)
    opts = {
        "Uxgz": init, "Uygz": init, "Uz": init, "Uxyz": init,
    }
    alpha = 0.5
    G = np.ones((var_num, var_num)) - np.diag([1] * var_num)
    var = list(range(0, len(data)))
    S = [[] for i in range(var_num)]
    for i in range(var_num):
        for j in range(var_num):
            S[i].append([])
    for i in range(len(var)):
        # print(i)
        if (neighboursize(G, var) < i).all():
            break
        for x in var:
            nb = neighbor(G, x)
            for y in nb:
                subsets = mynchoosek(list(set(nb) - set([y])), i)
                if len(subsets) == 0:

                    opts["Uxgz"] = alpha / (np.prod([nstates[x]]))
                    opts["Uygz"] = alpha / (np.prod([nstates[y]]))
                    opts["Uz"] = alpha / (1)
                    opts["Uxyz"] = alpha / (np.prod([nstates[x]]) * np.prod([nstates[y]]))
                    if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num), [],
                                    [nstates[x]],
                                    [nstates[y]], [], 0, opts)[0]:
                        G[x, y] = G[y, x] = 0
                else:
                    for j in range(len(subsets)):
                        # print(i, x, y, j)
                        z = subsets[j]
                        opts["Uxgz"] = alpha / (np.prod(nstates[x]) * np.prod(nstates[z]))
                        opts["Uygz"] = alpha / (np.prod(nstates[y]) * np.prod(nstates[z]))
                        opts["Uz"] = alpha / (np.prod(nstates[z]))
                        opts["Uxyz"] = alpha / (np.prod(nstates[x]) * np.prod(nstates[y]) * np.prod(nstates[z]))
                        if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                                        data[z, :].reshape(-1, data_num), nstates[x],
                                        nstates[y], nstates[z], 0, opts)[0]:
                            G[x, y] = G[y, x] = 0
                            S[x][y] = list(set(S[x][y]) | set(z))
                            S[y][x] = S[x][y]
    return G, S


def PCskletetonDataBlock(data, block):
    var_num = len(block)
    data_num = data.shape[1]
    nstates = np.max(data, 1).T
    nstates = nstates.astype(np.int)

    init = max(0.0002 * data_num, 1)
    opts = {
        "Uxgz": init, "Uygz": init, "Uz": init, "Uxyz": init,
    }
    alpha = 10
    G = np.ones((var_num, var_num)) - np.diag([1] * var_num)
    var = list(range(0, var_num))
    S = [[] for i in range(var_num)]
    for i in range(var_num):
        for j in range(var_num):
            S[i].append([])
    for i in range(len(var)):
        print(i)
        if (neighboursize(G, var) < i).all():
            break
        for tempx in var:
            nb = neighbor(G, tempx)
            for tempy in nb:
                subsets = mynchoosek(list(set(nb) - set([tempy])), i)
                if len(subsets) == 0:
                    x = block[tempx]
                    y = block[tempy]

                    opts["Uxgz"] = alpha / (np.prod([nstates[x]]))
                    opts["Uygz"] = alpha / (np.prod([nstates[y]]))
                    opts["Uz"] = alpha / (1)
                    opts["Uxyz"] = alpha / (np.prod([nstates[x]]) * np.prod([nstates[y]]))
                    if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num), [],
                                    nstates[x],
                                    nstates[y], [], 0, opts)[0]:
                        G[tempx, tempy] = G[tempy, tempx] = 0
                else:
                    x = block[tempx]
                    y = block[tempy]
                    for j in range(len(subsets)):
                        print(i, tempx, tempy, j)
                        tempz = subsets[j]
                        z = [block[sub] for sub in tempz]
                        z = sum(z, [])
                        opts["Uxgz"] = alpha / (np.prod(nstates[x]) * np.prod(nstates[z]))
                        opts["Uygz"] = alpha / (np.prod(nstates[y]) * np.prod(nstates[z]))
                        opts["Uz"] = alpha / (np.prod(nstates[z]))
                        opts["Uxyz"] = alpha / (np.prod(nstates[x]) * np.prod(nstates[y]) * np.prod(nstates[z]))
                        if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                                        data[z, :].reshape(-1, data_num), nstates[x],
                                        nstates[y], nstates[z], 0, opts)[0]:
                            G[tempx, tempy] = G[tempy, tempx] = 0
                            S[tempx][tempy] = list(set(S[tempx][tempy]) | set(tempz))
                            S[tempy][tempx] = S[tempx][tempy]
    return G, S


def PCorient(G, S):
    GpDAG = copy.deepcopy(G)
    var_num = G.shape[0]
    for x in range(var_num):
        for y in range(x + 1, var_num):
            G_X1 = GpDAG[x, :]
            G_X2 = GpDAG[:, x]
            n_x1 = np.where(G_X1 > 0)[0].tolist()
            n_x2 = np.where(G_X2 > 0)[0].tolist()
            undirected_neighbours_x = list(set(n_x1) & set(n_x2))
            G_Y1 = GpDAG[y, :]
            G_Y2 = GpDAG[:, y]
            n_y1 = np.where(G_Y1 > 0)[0].tolist()
            n_y2 = np.where(G_Y2 > 0)[0].tolist()
            undirected_neighbours_y = list(set(n_y1) & set(n_y2))
            cands = list(set(undirected_neighbours_x) & set(undirected_neighbours_y))
            for z in cands:
                if not (z in S[x][y]):
                    GpDAG[z, x] = 0
                    GpDAG[z, y] = 0
    return GpDAG


def PCskletetonData_neuron(data, condition):
    var_num = data.shape[0]
    data_num = data.shape[1]
    nstates = np.max(data, 1).T
    nstates = nstates.astype(np.int)
    opts = {
        "Uxgz": 0.1, "Uygz": 0.1, "Uz": 0.1, "Uxyz": 0.1,
    }
    G = np.ones((var_num, var_num)) - np.diag([1] * var_num)
    var = list(range(0, len(data)))
    S = [[] for i in range(var_num)]
    for i in range(var_num):
        for j in range(var_num):
            S[i].append([])
    for i in range(len(var)):
        if (neighboursize(G, var) < i).all():
            break
        for x in var:
            nb = neighbor(G, x)
            for y in nb:
                subsets = mynchoosek(list(set(nb) - set([y])), i)
                if len(subsets) == 0:
                    condition_nstates = np.array(np.max(condition, 1).T.astype(np.int).tolist())
                    if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num), condition,
                                    [nstates[x]],
                                    [nstates[y]], condition_nstates, 0, opts)[0]:
                        G[x, y] = G[y, x] = 0
                else:
                    for j in range(len(subsets)):
                        z = subsets[j]
                        condition_data = np.vstack((data[z, :].reshape(-1, data_num), condition))
                        condition_nstates = np.array(
                            nstates[z].tolist() + np.max(condition, 1).T.astype(np.int).tolist())
                        if condindepEmp(data[x, :].reshape(-1, data_num), data[y, :].reshape(-1, data_num),
                                        condition_data, nstates[x],
                                        nstates[y], condition_nstates, 0, opts)[0]:
                            G[x, y] = G[y, x] = 0
                            S[x][y] = list(set(S[x][y]) | set(z))
                            S[y][x] = S[x][y]
    return G, S
