import warnings

warnings.filterwarnings("ignore")
node_number = 3
import copy
import itertools
import numpy as np
import pandas as pd
from utility import generate_all_Markov_network  # , gibbs_sampling

import multiprocessing
from functools import partial
import pickle
import matplotlib.pyplot as plt

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel, MarkovNetwork
from pgmpy.sampling import GibbsSampling
from utility import *
from collections import Counter
from scipy.stats import entropy

networks, potentials = generate_all_Markov_network(node_number)


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


def getJointDistrubution_(threshold=0.005, joint_probs=None):
    flag = True
    G = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    )
    p = np.random.uniform(0, 1)
    # model = BayesianNetwork()
    # model.add_node(0)
    # model.add_node(1)
    # model.add_node(2)
    # if p <= 0.5:
    #     model.add_edge(u=0, v=1)
    #     model.add_edge(u=0, v=2)
    #     cpd1 = TabularCPD(0, 2, [[0.3], [0.7]])
    #     cpd2 = TabularCPD(1, 2, [[0.3, 0.2],
    #                              [0.7, 0.8]],
    #                       evidence=[0],
    #                       evidence_card=[2])
    #     cpd3 = TabularCPD(2, 2, [[0.6, 0.4],
    #                              [0.4, 0.6]],
    #                       evidence=[0],
    #                       evidence_card=[2])
    #     model.add_cpds(cpd1, cpd2, cpd3)
    # else:
    #     model.add_edge(u=0, v=1)
    #     model.add_edge(u=2, v=1)
    #     cpd1 = TabularCPD(0, 2, [[0.3], [0.7]])
    #     cpd2 = TabularCPD(2, 2, [[0.8], [0.2]])
    #     cpd3 = TabularCPD(1, 2, [[0.3, 0.2, 0.1, 0.6],
    #                              [0.7, 0.8, 0.9, 0.4]],
    #                       evidence=[0, 2],
    #                       evidence_card=[2, 2])
    #     model.add_cpds(cpd1, cpd2, cpd3)
    # model.add_cpds(cpd3, cpd1, cpd2)
    # bp = BeliefPropagation(model)
    # joint_probs = bp.query([0, 1, 2]).values
    # mModel = model.to_markov_model()
    # G = np.zeros((3, 3))
    # for edge in mModel.edges:
    #     G[edge[0]][edge[1]] = 1
    #     G[edge[1]][edge[0]] = 1
    # return BeliefPropagation(mModel).query([0, 1, 2]).values, G, flag
    if joint_probs is None:
        joint_probs = np.random.rand(2, 2, 2)
        joint_probs = joint_probs / joint_probs.sum()

    p0 = joint_probs.sum(axis=(1, 2))
    p1 = joint_probs.sum(axis=(0, 2))
    p2 = joint_probs.sum(axis=(0, 1))

    # P(0,1)
    p01 = joint_probs.sum(axis=(2))
    p_ind = np.outer(p0, p1)
    p_ind = p_ind / np.sum(p_ind)
    J01 = kl_divergence(p01.reshape(-1), p_ind.reshape(-1))

    p12 = joint_probs.sum(axis=(0))
    p_ind = np.outer(p1, p2)
    p_ind = p_ind / np.sum(p_ind)
    J12 = kl_divergence(p12.reshape(-1), p_ind.reshape(-1))

    p02 = joint_probs.sum(axis=(1))
    p_ind = np.outer(p0, p2)
    p_ind = p_ind / np.sum(p_ind)
    J02 = kl_divergence(p02.reshape(-1), p_ind.reshape(-1))

    ########################################################
    # P(0,1|2)
    p01_2 = np.zeros((2, 2, 2))
    for k in range(2):
        z = np.sum(joint_probs[:, :, k])
        for i in range(2):
            for j in range(2):
                p = joint_probs[i, j, k] / z
                p01_2[i, j, k] = p
    # P(0|2)
    p0_2 = np.sum(p01_2, axis=1)
    p1_2 = np.sum(p01_2, axis=0)
    # P(0|2)*P(1|2)
    p_indep = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = p0_2[i, k] * p1_2[j, k]
                p_indep[i, j, k] = p
    C01 = kl_divergence(p01_2.reshape(-1), p_indep.reshape(-1))

    ########################################################
    p12_0 = np.zeros((2, 2, 2))
    for i in range(2):
        z = np.sum(joint_probs[i, :, :])
        for j in range(2):
            for k in range(2):
                p = joint_probs[i, j, k] / z
                p12_0[i, j, k] = p
    # P(0|2)
    p1_0 = np.sum(p12_0, axis=2)
    p2_0 = np.sum(p12_0, axis=1)
    p_indep = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = p1_0[i, j] * p2_0[i, k]
                p_indep[i, j, k] = p
    C12 = kl_divergence(p12_0.reshape(-1), p_indep.reshape(-1))

    ########################################################
    p02_1 = np.zeros((2, 2, 2))
    for j in range(2):
        z = np.sum(joint_probs[:, j, :])
        for i in range(2):
            for k in range(2):
                p = joint_probs[i, j, k] / z
                p02_1[i, j, k] = p

    # P(0|1) P(2|1)
    p0_1 = np.sum(p02_1, axis=2)
    p2_1 = np.sum(p02_1, axis=0)
    p_indep = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = p0_1[i, j] * p2_1[j, k]
                p_indep[i, j, k] = p
    C02 = kl_divergence(p02_1.reshape(-1), p_indep.reshape(-1))

    if C01 < threshold or J01 < threshold:
        G[0, 1] = 0
        G[1, 0] = 0
    elif C01 > 0.2 and J01 > 0.2:
        G[0, 1] = 1
        G[1, 0] = 1
    else:
        flag = False

    if C12 < threshold or J12 < threshold:
        G[1, 2] = 0
        G[2, 1] = 0
    elif C12 > 0.2 and J12 > 0.2:
        G[1, 2] = 1
        G[2, 1] = 1
    else:
        flag = False

    if C02 < threshold or J02 < threshold:
        G[0, 2] = 0
        G[2, 0] = 0
    elif C02 > 0.2 and J02 > 0.2:
        G[0, 2] = 1
        G[2, 0] = 1
    else:
        flag = False

    return joint_probs, G, flag


def get_joint_probs(optential, n):
    # joint_probs = np.zeros((2, 2, 2))
    # combinations = []
    # for i in range(1, 4):
    #     com = list(itertools.combinations([0, 1, 2], i))
    #     combinations += com
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             state = [i, j, k]
    #             states = [[state[c] for c in co] for co in combinations]
    #             P = 1
    #             for c, co in enumerate(combinations):
    #                 if len(co) == 1:
    #                     co = co[0]
    #                 if co not in optential.keys():
    #                     continue
    #                 p = np.array(optential[co])[tuple(states[c])]
    #                 P *= p
    #             joint_probs[i, j, k] = P
    # joint_probs = joint_probs / joint_probs.sum()
    #
    # G = getJointDistrubution_(joint_probs=joint_probs, threshold=1e-5)
    # Create an empty joint probability distribution array with an initial value of 1
    joint_prob = np.ones((2,) * n)

    # For each node's potential function, multiply it by the joint probability distribution
    for node_ids, node_cpd in potential.items():
        joint_prob *= node_cpd

    return joint_prob


def gibbs_sampling(num_samples, graph):
    adj_matrix = graph[0]
    joint_probs = graph[1]
    # Determine whether it is independent
    names = []
    pros = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                name = [i, j, k]
                names.append(name)
                p = joint_probs[i, j, k]
                pros.append(p)

    samples = np.random.choice(a=list(range(len(names))), p=pros, size=num_samples)
    sampleData = (np.array([names[s] for s in samples], dtype=np.int64) + 1).T
    learned_graph = PC(sampleData, sampleData.shape[1])
    # if np.mean(learned_graph == adj_matrix) != 1:
    #     print(np.mean(learned_graph == adj_matrix))
    return joint_probs, learned_graph, sampleData


def simulation_Markov_data_par(graph, num_sample):
    #
    print(num_sample)
    joint_probs, learned_graph, sampleData = gibbs_sampling(num_sample, graph)
    return joint_probs, learned_graph, sampleData


def generate_simulation_Markov_data_par(num_sample, m):
    data = pd.read_pickle("./data/finalJustMarkovGraph.pkl")
    graphs = data["G"]
    joint_pros = data['joint_probs']

    new_graphs = []
    new_joint_pros = []
    for i in range(len(graphs)):
        for j in range(len(joint_pros[i])):
            new_graphs.append(deepcopy(graphs[i]))
            new_joint_pros.append(deepcopy(joint_pros[i][j]))
    # graph = np.random.randint(0, len(graphs), size=m)
    graph = np.random.randint(0, len(new_graphs), size=m)
    graphs = [(new_graphs[g], new_joint_pros[g]) for g in graph]
    # data = []
    # accuracy = []
    # for i in range(len(graphs)):
    #     # print(i, "=======")
    #     d = simulation_Markov_data_par(graphs[i], num_sample)
    #     a = np.array(d[1] == graphs[i][0], dtype=np.int64)
    #     a = (a[0, 1] + a[0, 2] + a[1, 2]) / 3
    #     accuracy.append(a)
    #     data.append(d)
    # print(np.mean(accuracy), np.std(accuracy))
    with multiprocessing.Pool(processes=20) as pool:
        data = pool.map(partial(simulation_Markov_data_par, num_sample=num_sample), graphs)

    # node_datas = [d[0] for d in data]
    joint_probs = [d[0] for d in data]
    G = [d[1] for d in data]
    node_datas = [d[2] for d in data]
    result = {
        "graphs": graphs,
        "node_datas": node_datas,
        "joint_probs": joint_probs,
        "G": G
    }
    name = str(num_sample)
    name = "0" * (4 - len(name)) + name
    print(name)
    with open("./data/MarkovJust/MNData" + name + ".pkl", "wb") as file:
        pickle.dump(result, file)


def generate_simulation_Markov_data(m=1000):
    num_samples = list(range(20, 2001, 20))
    for num_sample in num_samples:
        print(num_sample, "=" * 10)
        generate_simulation_Markov_data_par(num_sample=num_sample, m=m)


##############################################################################

if __name__ == '__main__':
    generate_simulation_Markov_data(m=50000)
