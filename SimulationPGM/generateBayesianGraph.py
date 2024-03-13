import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import pickle
from utility import generate_all_Markov_network, generate_all_belief_network
import copy
import itertools

node_number = 3
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

        # If there is an edge, it needs to satisfy it is  related.
        # If there is no edge, the condition needs to be satisfied and it is unrelated.

        uneighbor = np.where(upstream_graph[:, n] != 0)[0]
        if len(uneighbor) == 0:
            if n in neighbor and kl < 0.1:
                return True
            elif n not in neighbor and kl > 0.01:
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
            if n in neighbor and kl < 0.1:
                return True
            elif n not in neighbor and kl > 0.01:
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
        if NUM > 1000:
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


def generateRandomGraph(data):
    upstream_graph, upstream_joint_prob, graph = data

    CPDS = []
    while len(CPDS) < 10:
        print(len(CPDS))
        pro_table = generateCPD(upstream_num=3, downstream_num=2, graph=graph, upstream_joint_prob=upstream_joint_prob,
                                upstream_graph=upstream_graph)

        if pro_table is None:
            x = 0
        CPDS.append(pro_table)
        num = [c for c in CPDS if c is None]
        if len(num) == 10 and len(CPDS) == 10:
            break
    return CPDS


def Main(m=2000):
    data = pd.read_pickle("./data/MarkovGraph.pkl")
    joint_probs = data["joint_probs"]
    MarKovGraphs = data["G"]

    bayesianGraph = []
    bayesianGraphJoint = []
    for i in range(len(MarKovGraphs)):
        bayesianGraph.append([])
        bayesianGraphJoint.append([])
        for j in range(len(joint_probs[i])):
            print(i, j)
            index = np.random.randint(0, len(belief_network), size=1)[0]
            message = (MarKovGraphs[i], joint_probs[i][j], belief_network[index])
            d = generateRandomGraph(message)
            bayesianGraph[i].append(belief_network[index])
            bayesianGraphJoint[i].append(d)


    result = {
        "MarKovGraphs": MarKovGraphs,
        "joint_probs": joint_probs,
        "bayesianGraph": bayesianGraph,
        "bayesianGraphJoint": bayesianGraphJoint
    }

    with open("./data/bayesianGraph.pkl", "wb") as file:
        pickle.dump(result, file)


def selectedGraph():
    data = pd.read_pickle("./data/bayesianGraph.pkl")
    joint_probs = data["joint_probs"]
    bayesianGraph = data["bayesianGraph"]
    bayesianGraphJoint = data["bayesianGraphJoint"]

    newJointProbs = []
    newBayesianGraph = []
    newBayesianGraphJoint = []

    for i in range(len(bayesianGraphJoint)):
        index = [j for j in range(len(bayesianGraphJoint[i])) if
                 None not in bayesianGraphJoint[i][j]]

        # index = np.random.choice(a=index, p=[1 / len(index)] * len(index), size=10, replace=False)

        tempJointProbs = [joint_probs[i][j] for j in index]

        tempBayesianGraph = [bayesianGraph[i][j] for j in index]

        tempBayesianGraphJoint = [bayesianGraphJoint[i][j] for j in index]

        newJointProbs.append(tempJointProbs)

        newBayesianGraph.append(tempBayesianGraph)

        newBayesianGraphJoint.append(tempBayesianGraphJoint)

    result = {
        "G": data["MarKovGraphs"],
        "joint_probs": newJointProbs
    }
    with open("./data/finalMarkovGraph.pkl", "wb") as file:
        pickle.dump(result, file)

    result = {
        "G": data["MarKovGraphs"],
        "joint_probs": newJointProbs,
        "BayesianGraph": newBayesianGraph,
        "BayesianGraphJoint": newBayesianGraphJoint,
    }
    with open("./data/finalBayesianGraph.pkl", "wb") as file:
        pickle.dump(result, file)


if __name__ == '__main__':
    # selectedGraph()
    Main()
