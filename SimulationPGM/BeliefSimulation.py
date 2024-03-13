import copy
import itertools
import numpy as np
import pandas as pd
from utility import generate_all_Markov_network, generate_all_belief_network
from pgmpy.models import BayesianNetwork
import multiprocessing
from functools import partial
import pickle
import matplotlib.pyplot as plt
from pgmpy.sampling import BayesianModelSampling
import warnings
from pgmpy.factors.discrete import State
from pgmpy.factors.discrete import TabularCPD
from itertools import product
from bayesianScore import learnBayesNetBlock
import os

warnings.filterwarnings("ignore")
node_number = 3
# networks = generate_all_Markov_network(node_number)

downstream_number = 2
belief_network = generate_all_belief_network(node_number, downstream_number)


def kl_divergence(P, Q):
    """
    Compute the KL divergence D(P || Q) for discrete distributions
    :param P: np.array, first probability distribution
    :param Q: np.array, second probability distribution
    :return: float, KL divergence of P and Q
    """
    # Ensure the probability distributions are normalized
    P = P / P.sum()
    Q = Q / Q.sum()
    # Compute the KL divergence, taking care not to divide or take the logarithm of zero
    return np.sum(P * (np.log2(P) - np.log2(Q)))


def generate_dict_cpd(child, parents):
    cpd_dict = {}
    cpds = []
    for i in parents:
        p = np.random.uniform()
        if p <= 0.5:
            p = np.random.uniform(0, 0.3)
        else:
            p = np.random.uniform(0.7, 1)
        p = np.array([[p, 1 - p], [1 - p, p]])
        cpds.append(p)

    def deep(i, n, state, joint):
        if i == n:
            p = np.array([1, 1])
            for j in range(len(state)):
                p = p * cpds[j][tuple([state[j]])]
            p = p / np.sum(p)
            joint[tuple(state + [0])] = p[0]
            joint[tuple(state + [1])] = p[1]
        else:
            for s in range(2):
                deep(i + 1, n, state + [s], joint)

    if len(parents) != 0:
        cpd = np.zeros((2,) * (len(parents) + 1))
        deep(0, len(parents), [], cpd)
    else:
        p = np.random.uniform()
        if p <= 0.5:
            p = np.random.uniform(0, 0.3)
        else:
            p = np.random.uniform(0.7, 1)
        cpd = np.array([p, 1 - p])

    return cpd


def getCondition(G):
    """
   According to the graph structure, when judging each node as a parent node, a condtion node is required.
    :return:
    """
    condition = []
    for i in range(len(G)):
        nondes = list(np.where(G[i, :] == 1)[0])
        condition.append(nondes)
    return condition


def jointMutiCPD(i, joint, cpd, n, state, joint_distrubution, neighbor):
    if i == n:
        joint_pro = joint[tuple(state)]
        cpd_pro = cpd[tuple(np.array(state)[neighbor])]
        p = joint_pro * cpd_pro
        axis = tuple(state + [0])
        joint_distrubution[axis] = p[0]
        axis = tuple(state + [1])
        joint_distrubution[axis] = p[1]
    else:
        for s in range(2):
            x = 0
            jointMutiCPD(i + 1, joint, cpd, n, state + [s], joint_distrubution, neighbor)
            x = 0


def CPD(i, n, joint, cpd, cpd_var, state):
    if i == n:
        num = len([s for s in state if s is None])
        combinations = list(itertools.product([0, 1], repeat=num))
        new_states = []
        Z = 0
        for combination in combinations:
            cnt = 0
            new_state = copy.deepcopy(state)
            for j in range(len(new_state)):
                if new_state[j] is None:
                    new_state[j] = combination[cnt]
                    cnt += 1
            new_states.append(new_state)
            p = joint[tuple(new_state)]
            Z += p
        for new_state in new_states:
            cpd[tuple(new_state)] = joint[tuple(new_state)] / Z
        x = 0

    else:
        if i in cpd_var:
            for s in range(2):
                CPD(i + 1, n, joint, cpd, cpd_var, state + [s])
        else:
            CPD(i + 1, n, joint, cpd, cpd_var, state + [None])


def mutiCPD(i, t, pncu, pxcu, x, n, pro, condition, state):
    if i == t:
        state1 = []
        for j in range(len(state)):
            if j == n or j in condition:
                state1.append(state[j])
        p1 = pncu[tuple(state1)]
        state2 = []
        for j in range(len(state)):
            if j == x or j in condition:
                state2.append(state[j])
        p2 = pxcu[tuple(state2)]
        p = p1 * p2
        state3 = []
        for j in range(len(state)):
            if j == n or j == x or j in condition:
                state3.append(state[j])
        pro[tuple(state3)] = p

    else:
        tempState = list(condition) + [x] + [n]
        tempState.sort()
        if i in tempState:
            for s in range(2):
                mutiCPD(i + 1, t, pncu, pxcu, x, n, pro, condition, state + [s])
        else:
            mutiCPD(i + 1, t, pncu, pxcu, x, n, pro, condition, state + [None])


def isIndepent(node, upstream_joint_prob, pro_table, graph, upstream_graph):
    #
    neighbor = np.where(graph[:, node] == 1)[0]
    neighbor = list(neighbor)
    axis = [0, 1, 2]
    for n in neighbor:
        axis.remove(n)
    # parent_joint_pro = np.sum(upstream_joint_prob, axis=tuple(axis))
    cpd_pro = pro_table
    joint_child_parent_pro = np.zeros((2,) * (4))
    jointMutiCPD(0, upstream_joint_prob, cpd_pro, 3, [], joint_child_parent_pro, neighbor)

    for n in range(joint_child_parent_pro.ndim - 1):
        axis = list(range(joint_child_parent_pro.ndim))
        axis.remove(n)
        # P(n)
        p0 = np.sum(joint_child_parent_pro, axis=tuple(axis))

        # P(x)
        axis = list(range(joint_child_parent_pro.ndim))[:-1]
        px = np.sum(joint_child_parent_pro, axis=tuple(axis))

        # P(n,X)
        axis = list(range(joint_child_parent_pro.ndim))[:-1]
        axis.remove(n)
        px0 = np.sum(joint_child_parent_pro, axis=tuple(axis))

        # P(x)*P(n)
        p_index = p0.reshape(-1, 1) * px.reshape(1, -1)

        kl = kl_divergence(px0.reshape(-1), p_index.reshape(-1))

        if n in neighbor and kl < 0.1:
            return True
        elif n not in neighbor:
            uneighbor = np.where(upstream_graph[:, n] != 0)[0]
            if len(uneighbor) == 0 and kl > 0.01:
                return True
            elif len(uneighbor) != 0:
                # P(other|uneighbor)
                condition = list(uneighbor)
                pcond = np.zeros((2,) * 4)
                CPD(0, 4, joint_child_parent_pro, pcond, condition, [])
                # P(x|uneighbor)
                axis = list(range(joint_child_parent_pro.ndim))[:-1]
                for un in uneighbor:
                    axis.remove(un)
                pxcu = np.sum(pcond, axis=tuple(axis))

                # P(n|uneighbor)
                axis = list(range(joint_child_parent_pro.ndim))
                for un in uneighbor:
                    axis.remove(un)
                axis.remove(n)
                pncu = np.sum(pcond, axis=tuple(axis))

                #
                axis = list(range(joint_child_parent_pro.ndim))[:-1]
                for un in uneighbor:
                    axis.remove(un)
                axis.remove(n)
                pxncu = np.sum(pcond, axis=tuple(axis))

                pro = np.zeros((2,) * pxncu.ndim)
                mutiCPD(0, 4, pncu, pxcu, 3, n, pro, condition, [])
                kl = kl_divergence(pxncu.reshape(-1), pro.reshape(-1))
                if np.abs(kl) > 0.01:
                    return True
        # P(n|other)
        #
        # elif n not in neighbor and kl > 0.01:
        #     return True
    return False


def generateCPD(upstream_num, downstream_num, graph, upstream_joint_prob, upstream_graph):
    flag = False
    need = [True, True]
    pro_table = [None, None]
    NUM = 0
    while flag == False:
        if NUM > 5000:
            return None
        NUM += 1
        for i in range(upstream_num, upstream_num + downstream_num):
            if need[i - 3] == False:
                continue
            parents = list(np.where(graph[:, i] == 1)[0])
            pro = generate_dict_cpd(i, parents)
            pro_table[i - 3] = pro

        indepent1 = isIndepent(3, upstream_joint_prob, pro_table[0], graph, upstream_graph)
        if indepent1 == False:
            need[0] = False
        indepent2 = isIndepent(4, upstream_joint_prob, pro_table[1], graph, upstream_graph)
        if indepent2 == False:
            need[1] = False
        if indepent1 == False and indepent2 == False:
            flag = True
    return pro_table


def gibbs_sampling(upstream_graph, upstream_data, graph, num_sample, upstream_joint_prob, pro_table):
    downstream_num = len(graph) - len(upstream_graph)
    upstream_num = len(upstream_graph)

    # pro_table = generateCPD(upstream_num, downstream_num, graph, upstream_joint_prob, upstream_graph)
    # if pro_table is None:
    #     return None

    sample = np.zeros((len(graph), num_sample), dtype=np.int64)
    for i in range(upstream_data.shape[1]):
        for j in range(upstream_num, upstream_num + downstream_num):
            parent = np.where(graph[:, j] == 1)[0]
            state = list(np.array(list(upstream_data[parent, i])) - 1)
            try:
                p = pro_table[j - 3][tuple(state)]
                sample[j, i] = int(np.random.choice(a=[0, 1], p=p, size=1)[0] + 1)
            except:
                print("error=====" * 100)
    for i in range(upstream_num):
        sample[i, :] = upstream_data[i, :]
    data = sample
    blockMessage = [[i] for i in range(upstream_num)]
    condition = getCondition(upstream_graph)
    nstates = np.max(data, axis=1).T
    nstates = np.array(nstates, dtype=np.int64)
    Alearn, bestparameters, bestparents, bestscores = learnBayesNetBlock(data=data, nstates=nstates,
                                                                         blockMessage=blockMessage,
                                                                         casualNum=upstream_num,
                                                                         blockNum=upstream_num,
                                                                         effectNum=downstream_num, conditions=condition)
    # print(np.mean(Alearn[:, -2:] == graph[:, -2:]))
    return Alearn


def simulation_bayesian_data_par(data, num_sample):
    print(num_sample)
    upstream_data = data[0]
    upstream_graph = data[1]
    joint = data[2]
    graph = data[3]
    cpd = data[4]
    learned_graph = gibbs_sampling(upstream_graph, upstream_data, graph, num_sample, joint, cpd)
    return learned_graph


def generate_simulation_bayesian_data_par(data):
    num_sample = data[1]
    upstream_data = data[0]
    upstream_graphs = upstream_data["graphs"]
    node_datas = upstream_data["node_datas"]
    upstream_joint = upstream_data["joint_probs"]
    m = len(upstream_graphs)

    message = pd.read_pickle("./data/finalBayesianGraph.pkl")
    MG = message["G"]
    MJ = message["joint_probs"]
    BG = message["BayesianGraph"]
    BCPD = message["BayesianGraphJoint"]

    graphs = []
    cpds = []
    for upstream_graph in upstream_graphs:
        G = upstream_graph[0]
        J = upstream_graph[1]

        mgIndex = [np.mean(G == g) for g in MG]
        mgIndex = mgIndex.index(1)

        mj = MJ[mgIndex]
        mjIndex = [np.mean(J == j) for j in mj]
        mjIndex = mjIndex.index(1)

        bg = BG[mgIndex][mjIndex]
        bc = BCPD[mgIndex][mjIndex][0]

        graphs.append(copy.deepcopy(bg))
        cpds.append(copy.deepcopy(bc))

    tempData = [(node_datas[i], upstream_graphs[i][0], upstream_joint[i], graphs[i], cpds[i]) for i in range(m)]
    # for i in range(len(tempData)):
    #     simulation_bayesian_data_par(tempData[i], num_sample)
    with multiprocessing.Pool(processes=20) as pool:
        data = pool.map(partial(simulation_bayesian_data_par, num_sample=num_sample), tempData)

    # node_datas = [d[0] for d in data]
    G = [d for d in data]
    result = {
        "upstream_graphs": upstream_graphs,
        "graphs": graphs,
        # "node_datas": node_datas,
        "G": G
    }
    name = str(num_sample)
    name = "0" * (4 - len(name)) + name
    with open("./data/Bayesian/BNData" + name + ".pkl", "wb") as file:
        pickle.dump(result, file)


def generate_simulation_bayesian_data():
    # datas = pd.read_pickle("./data/MNData.pkl")
    # num_samples = list(datas.keys())
    # datas = [(datas[n], n) for n in num_samples]

    path = "./data/Markov/"
    fileNames = os.listdir(path)

    filePaths = [path + f for f in fileNames]
    filePaths.sort()
    filePaths = filePaths
    print(len(filePaths))
    for filepath in filePaths:
        print(filepath)
        d = pd.read_pickle(filepath)
        n = int(filepath.split("/")[-1][6:][:-4])
        generate_simulation_bayesian_data_par((d, n))


##############################################################################

if __name__ == '__main__':
    generate_simulation_bayesian_data()
