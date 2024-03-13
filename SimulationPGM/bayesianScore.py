import sys

sys.path.append("../")
import numpy as np
from Utils import *
from scipy.special import gammaln
from itertools import combinations
import random
import copy
import pandas as pd
from multiprocessing.dummy import Pool
from functools import partial
from condindepEmp import *
import multiprocessing


def data_balance(dataV, dataParents):
    num1 = np.sum(dataV - 1)
    num2 = len(dataV) - num1
    num = np.min([num1, num2])

    index1 = np.where(dataV == 2)[0].tolist()
    index2 = np.where(dataV == 1)[0].tolist()
    index = np.random.choice(index1, p=[1 / len(index1)] * len(index1), size=num).tolist() + \
            np.random.choice(index2, p=[1 / len(index2)] * len(index2), size=num).tolist()
    random.shuffle(index)
    return np.array(dataV)[index], np.array(dataParents)[:, index]


def BDscore(dataV, dataParents, nstatesV, nstatesPa, u):
    U = u * np.ones((nstatesV, int(np.prod(nstatesPa))))
    if len(dataParents) != 0:
        cvpa = count(np.vstack((dataV, dataParents)), np.hstack((nstatesV, nstatesPa)))
    else:
        if len(dataV.shape) == 2:
            t = max(dataV.shape[0], dataV.shape[1])
            cvpa = count(dataV.reshape(-1, t), np.array([nstatesV]))

        else:
            cvpa = count(dataV.reshape(-1, len(dataV)), np.array([nstatesV]))
    CvgPa = cvpa.reshape(np.prod(nstatesV), int(np.prod(nstatesPa)), order='F')
    Up = U + CvgPa

    score = np.sum(
        gammaln(np.sum(U, axis=0)) - gammaln(np.sum(Up, axis=0)) + np.sum(gammaln(Up), axis=0) - np.sum(gammaln(U),
                                                                                                        axis=0), axis=0)
    return score, Up


G_ = 0
end = -1
rules = []
rule = []
Time_step = 0
strategy_nums = 4


def deep_rule(s, deep):
    if deep == Time_step - 1:
        if G_[s, end + strategy_nums] != 0:
            rule.append(end)
            rules.append(copy.deepcopy(rule))
            rule.pop()
            return
    else:
        nexts = np.nonzero(G_[s, :])[0].tolist()
        for n in nexts:
            rule.append(n - strategy_nums)
            deep_rule(n - strategy_nums, deep + 1)
            rule.pop()
    return


def learnBayesNet_Option(data, nstates, U, time_step, strategy_num, pro, result=None):
    """
   Obtain the BDscore of two or more strategy combinations, filter connections based on the BDscore, and construct a graph structure
    :param data: 
    :param nstates: 
    :param U: 
    :param time_step: 
    :param strategy_num: 
    :param pro: 
    :param result: 
    :return: 
    """

    if strategy_num == 5:
        strategy_dict = {
            0: "G", 1: "L", 2: "e", 3: "A", 4: "E"
        }
    global G_, Time_step, end, rule, rules, strategy_nums
    strategy_nums = strategy_num
    Time_step = time_step
    if time_step > 1:
        G_ = result["G"]

    data = data[:strategy_num * (time_step + 1)]
    var_num = data.shape[0]
    strategies = list(range(strategy_num))


    bestparents = [[] for i in range(var_num)]
    scores = {}
    bestParams = {}
    ratios = {}

    for v in strategies:

        if time_step == 1:
            for c in strategies:
                if c == v:
                    continue
                # dataV, dataParents = data_balance(data[v + strategy_num, :], data[c, :].reshape(1, -1))
                dataV = data[v + strategy_num, :]
                dataParents = data[c, :]
                s_, u_ = BDscore(dataV, dataParents, nstates[v + strategy_num], nstates[c], U)
                s = BDscore(dataV, [], nstates[v + strategy_num], [], U)[0]
                u_ = u_ / np.sum(u_)
                if s_ > s and u_[1][1] > pro["one_pro"][strategy_dict[c]] * pro["one_pro"][
                    strategy_dict[v]] and s / s_ > 1:
                    scores.update({str(v) + "-" + str(c): s_ / len(dataV)})
                    bestParams.update({str(v) + "-" + str(c): u_})
                    ratios.update({str(v) + "-" + str(c): s / s_})
                    bestparents[v + strategy_num] = list(set(bestparents[v + strategy_num] + [c]))
            end = copy.deepcopy(v)
            for c in strategies:
                rule = []
                rules = []
                rule.append(c)
                deep_rule(c, 0)
                if [0, 1, 0] in rules:
                    x = 0
                for rule in rules:
                    temp_rule = [i * strategy_num + rule[i] for i in range(len(rule))]
                    dataV = data[temp_rule[-1], :]
                    dataParents = [[]] * len(rule[:-1])
                    for p in range(len(temp_rule[:-1])):
                        dataParents[p] = data[temp_rule[p], :]
                    dataParents = np.array(dataParents)
                    # dataV, dataParents = data_balance(dataV, dataParents.reshape(-1, data.shape[1]))
                    s_, u_ = BDscore(dataV, dataParents, nstates[v], nstates[temp_rule[:-1]], U)
                    u_ = u_ / np.sum(u_)
                    tem_rule = np.array(rule)[::-1].tolist()
                    tem_rule = [str(t) for t in tem_rule]
                    temp = "-".join(tem_rule[:-1])
                    if result["scores"].__contains__(temp):
                        # s = result[time_step - 2]["scores"][temp]
                        s = BDscore(dataV, dataParents[1:], nstates[v], nstates[rule[1:-1]], U)[0]
                        if s_ >= s and u_[-1, -1] > 0.1:
                            scores.update({temp + "-" + str(c): s_ / len(dataV)})
                            ratios.update({temp + "-" + str(c): s / s_})
                            bestParams.update({temp + "-" + str(c): u_})
                            bestparents[v + strategy_num] = list(set(bestparents[v + strategy_num] + [c]))
    Alearn = np.zeros((2 * strategy_num, 2 * strategy_num))
    for i in range(strategy_num, 2 * strategy_num):
        Alearn[bestparents[i], i] = 1
    return Alearn, bestparents, scores, bestParams, ratios


def BDscore_parallelize(Pa, v, data, nstates, U):
    bd, Up = BDscore(data[v, :], data[Pa, :], nstates[v], nstates[Pa], U)
    return Pa, bd, Up


def parentset_parallelize(nparents, var_casual):
    parentset = list(combinations(var_casual, nparents))
    parentset = [list(o) for o in parentset]
    return parentset


def learnBayesNet(data, nstates, maxNparents, casual_num=9, effect_num=7, U=1):
    var_num = data.shape[0]
    bestparents = [[] for i in range(var_num)]
    bestparameters = [[] for i in range(var_num)]
    bestscores = [[] for i in range(var_num)]
    var_casual = list(range(0, casual_num))
    var_effect = list(range(casual_num, casual_num + effect_num))
    for vind in range(len(var_effect)):
        v = var_effect[vind]
        bestscore, Up = BDscore(data[v, :], [], nstates[v], [], U)
        with multiprocessing.Pool(processes=12) as pool:
            parentsets = pool.map(partial(parentset_parallelize, var_casual=var_casual),
                                  list(range(1, maxNparents[v] + 1)))
        new_parentsets = []
        for p in parentsets:
            new_parentsets += p
        with multiprocessing.Pool(processes=12) as pool:
            Parents, Scores, Parameters = zip(
                *pool.map(partial(BDscore_parallelize, v=v, data=data, nstates=nstates, U=U), new_parentsets))
        Scores = np.array(Scores)
        index = np.where(Scores == np.max(Scores))[0][0]
        if Scores[index] < bestscore:
            bestparents[v] = []
            bestscores[v] = bestscore
            bestparameters[v] = Up
        else:
            bestparents[v] = Parents[index]
            bestscores[v] = bestscore / Scores[index]
            bestparameters[v] = Parameters[index]
    Alearn = np.zeros((var_num, var_num))
    for i in range(var_num):
        Alearn[bestparents[i], i] = 1
    return Alearn, bestparameters, bestparents, bestscores


def learnBayesNet_f(data, nstates, maxNparents, casual_num=9, effect_num=7, U=1):
    var_num = data.shape[0]
    bestparents = [[] for i in range(var_num)]
    bestparameters = [[] for i in range(var_num)]
    bestscores = [[] for i in range(var_num)]
    var_casual = list(range(0, casual_num))
    var_effect = list(range(casual_num, casual_num + effect_num))
    for vind in range(len(var_effect)):
        v = var_effect[vind]
        bestscore, Up = BDscore(data[v, :], [], nstates[v], [], U)
        new_parentsets = []
        for nparent in range(1, maxNparents[v] + 1):
            parentset = list(combinations(var_casual, nparent))
            parentset = [list(o) for o in parentset]
            new_parentsets += parentset
        for Pa in new_parentsets:
            bd, Up = BDscore(data[v, :], data[Pa, :], nstates[v], nstates[Pa], U)
            if bd > bestscore:
                bestparents[v] = Pa
                bestscores[v] = bd
                bestparameters[v] = Up
    Alearn = np.zeros((var_num, var_num))
    for i in range(var_num):
        Alearn[bestparents[i], i] = 1
    return Alearn, bestparameters, bestparents, bestscores


def learnBayesNet_noparallelize(data, nstates, maxNparents, casual_num=9, effect_num=7, U=1):
    var_num = data.shape[0]
    bestparents = [[] for i in range(var_num)]
    bestparameters = [[] for i in range(var_num)]
    bestscores = [[] for i in range(var_num)]
    var_casual = list(range(0, casual_num))
    var_effect = list(range(casual_num, casual_num + effect_num))
    for vind in range(len(var_effect)):
        v = var_effect[vind]
        bestscore, Up = BDscore(data[v, :], [], nstates[v], [], U)

        parentsets = []
        for i in list(range(1, maxNparents[v] + 1)):
            parentset = list(combinations(var_casual, i))
            parentset = [list(o) for o in parentset]
            parentsets.append(parentset)
        new_parentsets = []
        for p in parentsets:
            new_parentsets += p
        Parents = []
        Scores = []
        Parameters = []
        for Pa in new_parentsets:
            bd, Up = BDscore(data[v, :], data[Pa, :], nstates[v], nstates[Pa], U)
            Parents.append(Pa)
            Scores.append(bd)
            Parameters.append(Up)
        Scores = np.array(Scores)
        index = np.where(Scores == np.max(Scores))[0][0]
        if Scores[index] < bestscore:
            bestparents[v] = []
            bestscores[v] = bestscore
            bestparameters[v] = Up
        else:
            bestparents[v] = Parents[index]
            bestscores[v] = bestscore / Scores[index]
            bestparameters[v] = Parameters[index]
    Alearn = np.zeros((var_num, var_num))
    for i in range(var_num):
        Alearn[bestparents[i], i] = 1
    return Alearn, bestparameters, bestparents, bestscores




def learnBayesNetBlock(data, nstates, blockMessage, casualNum=7, blockNum=3, effectNum=7, alpha=0.5, conditions=None):
    var_num = data.shape[0]
    bestparents = [[] for i in range(var_num)]
    bestparameters = [[] for i in range(var_num)]
    bestscores = [[] for i in range(var_num)]
    var_casual = list(range(blockNum))
    var_effect = list(range(var_num - effectNum, var_num))
    Alearn = np.zeros((blockNum + effectNum, blockNum + effectNum))
    for vind in range(len(var_effect)):
        v = var_effect[vind]
        for pa in var_casual:
            condition = conditions[pa]
            condition = [blockMessage[c] for c in condition]
            condition = sum(condition, [])
            U = alpha   / (np.prod(nstates[v]) * np.prod(nstates[condition]))
            bd1, Up1 = BDscore(data[v, :], data[condition, :], nstates[v], nstates[condition], U)

            Pa = blockMessage[pa]
            Pa = condition + Pa
            U = alpha   / (np.prod(nstates[v]) * np.prod(nstates[Pa]))
            bd2, Up2 = BDscore(data[v, :], data[Pa, :], nstates[v], nstates[Pa], U)
            if bd1 / bd2 > 1:
                # print(pa, v, bd1 / bd2, bd2 - bd1, np.exp(bd2 - bd1))
                Alearn[pa, v - (casualNum - blockNum)] = 1
    return Alearn, bestparameters, bestparents, bestscores
