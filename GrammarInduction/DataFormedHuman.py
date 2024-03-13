
# Encoding=utf8
import pandas as pd
import pickle as pickle
import numpy as np
import copy
from PGM.Utils import *
import os
from copy import deepcopy
import warnings
from functools import partial
import multiprocessing

warnings.filterwarnings("ignore")
from utility import K, nearestNeighbors, toOnehot


def keepFirstPoint(data, strategyName):
    """
    For each continuous strategy, select the first strategy point and delete other points.
    :param data:
    :param strategyName:
    :return:
    """
    data["game"] = data.file.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )
    newData = pd.DataFrame()
    for idx, grp in data.groupby("game"):
        first_point_mask = (grp[strategyName] == grp[strategyName].shift(1)).sum(1) < len(
            strategyName
        )
        tempData = grp.loc[first_point_mask]
        newData = newData.append(deepcopy(tempData))
    newData = newData.sort_index()
    newData.drop(columns="game", inplace=True)
    newData.reset_index(drop=True, inplace=True)

    return newData


def combineEvade(x):
    x = list(x)
    if 2 in x:
        return 2
    else:
        return 1


def formStrategyToOnehot(data):
    """
    Convert the strategy column into onehot form. Each column represents whether a strategy is enabled or not.
    :param data:
    :return:
    """

    data.reset_index(drop=True, inplace=True)
    strategyNames = ["global", "local", "evade_blinky", "evade_clyde", "evade_3", "evade_4", "approach", "energizer",
                     "no_energizer", "vague", "stay"]
    strategies = np.stack(data["strategy"].apply(lambda x: toOnehot(x, len(strategyNames))))
    encodedStrategies = (
            pd.DataFrame(strategies == strategies.max(1, keepdims=True), columns=strategyNames).astype(int) + 1)

    data[strategyNames] = encodedStrategies
    # add dead strategy
    trials = list(data.groupby("file"))
    dead = [1 if trials[t][0].split("-")[0] == trials[t - 1][0].split("-")[0] else 0 for t in range(1, len(trials))]
    # 合并evade
    data["evade"] = data[["evade_blinky", "evade_clyde", "evade_3", "evade_4"]].apply(lambda x: combineEvade(x), axis=1)
    return data


def formData(path):
    """
    Preliminary conversion of data: 1. Convert strategy to one hot form 2. Select the first point with continuous strategy
    :param path:
    :param Type:
    :return:
    """
    strategyNames = ["global", "local", "evade_blinky", "evade_clyde", "approach",
                     "energizer", "no_energizer", "vague", "stay"]
    fileName = path[0]
    savePath = path[1]
    data = pd.read_pickle(fileName)
    data = formStrategyToOnehot(data)
    data = keepFirstPoint(data, strategyNames)
    data.to_pickle(savePath)


def formDataMain(date):
    """
    Change the data format and strategy to one hot format, and only select the first of two consecutive strategies.
    :return:
    """
    fileFolder = "../HumanData/DiscreteFeatureData/" + date + "/"
    saveFolder = "../HumanData/FormedData/" + date + "/"
    fileNames = os.listdir(fileFolder)

    filePaths = [(fileFolder + fileName, saveFolder + fileName) for fileName in fileNames]
    with multiprocessing.Pool(processes=12) as pool:
        pool.map(partial(formData), filePaths)


def consolidateDataATNearestNb(date):
    """
    Consolidate people's data together
    :return:
    """
    fileFolder = "../HumanData/FrameData/" + date + "/"
    neighbors, fileNames = nearestNeighbors(fileFolder="../HumanData/FrameData/")

    stateNames = ['IS1', 'IS2', 'PG1', 'PG2', 'PE', 'BN5', 'BN10', "PA0", "PA1", "PA2", "PA3", "PA4", "PA5", "GA0",
                  "GA1", "GA2", "GA3", "GA4", "GA5"]
    strategyNames = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer",
                     "no_energizer", "stay", "vague"]
    S = ["G", "L", "1", "2", "A", "E", "N", "S", "V"]
    saveFolder = "../HumanData/seq/" + date + "/"
    numTolabel = {0: "G", 1: "L", 2: "1", 3: "2", 4: "3", 5: "4", 6: "A", 7: "E", 8: "N", 9: "V", 10: "S"}

    for k in range(len(neighbors)):
        index = neighbors[k]
        fileNames = os.listdir(fileFolder)
        fileNames = list(np.array(fileNames)[index])
        data = []
        points = [-1]
        for fileName in fileNames:
            temp_data = pd.read_pickle(fileFolder + fileName)
            points.append(points[-1] + len(temp_data))
            data.append(copy.deepcopy(temp_data))
        data = pd.concat(data, axis=0)

        states = data[stateNames]
        strategy = data[strategyNames]
        strategyLabel = data["strategy"].apply(lambda x: numTolabel[x])

        seq = ""
        seqs = []
        t = ""
        points = points[1:]
        for i in range(len(data)):
            temp = data[strategyNames].iloc[i]
            temp = np.argmax(temp)
            seq += S[temp]
            t += S[temp]
            if i - 1 in points:
                seqs.append(t)
                t = ""

        result = {
            "seq": seq,
            "S": S,
            "state": states,
            "strategy": strategy,
            "strategyLabel": strategyLabel,
            "fileNames": fileNames,
        }
        with open(saveFolder + fileNames[0], "wb") as file:
            pickle.dump(result, file)


def consolidateHumanGramDepth(K=2, only_strategy=False):
    """
     Consolidate people's data together
    :return:
    """
    # seq_cluster
    fileFolder = "../HumanData/FrameData/"
    allFileNames = os.listdir(fileFolder)
    AmateurIndex = [0, 9, 14, 16, 24, 26, 32]
    labels = np.array([0] * len(allFileNames))
    labels[AmateurIndex] = 1

    names = ["Novice", "Expert"]
    for k in range(K):
        index = np.where(labels == k)[0]
        fileNames = list(np.array(allFileNames)[index])
        data = []
        points = [-1]
        for fileName in fileNames:
            temp_data = pd.read_pickle(fileFolder + fileName)
            points.append(points[-1] + len(temp_data))
            data.append(copy.deepcopy(temp_data))
        data = pd.concat(data, axis=0)

        states_keys = ['IS', 'IS1', 'IS2', 'PE', 'PG', 'PG1', 'PG2', 'BN5', 'BN10', "PA0", "PA1",
                       "PA2", "PA3", "PA4", "PA5", "GA0",
                       "GA1", "GA2", "GA3", "GA4", "GA5"]
        strategy_keys = ["global", "local", "evade", "evade_blinky", "evade_clyde", "approach", "energizer",
                         "no_energizer", "stay",
                         "vague"]
        states = data[states_keys]
        strategy = data[strategy_keys]
        numTolabel = {0: "G", 1: "L", 2: "1", 3: "2", 4: "3", 5: "4", 6: "A", 7: "E", 8: "N", 9: "V", 10: "S"}
        strategyLabel = data["strategy"].apply(lambda x: numTolabel[x])

        S = ["G", "L", "1", "2", "A", "E", "N", "S", "V"]
        seq = ""
        seqs = []
        t = ""
        points = points[1:]
        for i in range(len(data)):
            temp = \
                data[['global', 'local', "evade_blinky", "evade_clyde", 'approach', 'energizer', "no_energizer", "stay",
                      "vague"]].iloc[
                    i]
            tttemp = np.where(np.array(temp) == 2)[0]
            if len(tttemp) > 1:
                print("=" * 100)
            temp = np.argmax(temp)
            seq += S[temp]
            t += S[temp]
            if i - 1 in points:
                seqs.append(t)
                t = ""

        result = {
            "seq": seq,
            "S": S,
            "state": states,
            "strategy": strategy,
            "strategyLabel": strategyLabel,
            "fileNames": fileNames
        }

        print(len(states))
        with open("../HumanData/seq_cluster/" + names[k] + ".pkl", "wb") as file:
            pickle.dump(result, file)


def form_date_human(date):
    formDataMain(date)
    consolidateDataATNearestNb(date)


if __name__ == '__main__':
    date = "session2"
    form_date_human(date)
