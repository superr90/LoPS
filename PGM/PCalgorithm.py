import numpy as np
from PGM.Utils import *
from PGM.condindepEmp import *
import copy

def PCskletetonData(data, alpha=0.5):
    var_num = data.shape[0]
    data_num = data.shape[1]
    nstates = np.max(data, 1).T
    nstates = nstates.astype(np.int64)
    init = max(0.0002 * data_num, 1)
    opts = {
        "Uxgz": init, "Uygz": init, "Uz": init, "Uxyz": init,
    }
    # alpha = 0.1
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

