import copy

import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing
from functools import partial
import scipy.stats as stats


def grubbs_test(data, alpha=0.05):
    """
    Grubbs' test, detect and remove outliers

     parameter:
     - data: one-dimensional array, containing original data
     - alpha: significance level, usually 0.05

     return value:
     - Cleaned data, excluding outliers
    """
    n = len(data)
    is_outlier = True
    while is_outlier:
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        max_idx = np.argmax(np.abs(data - mean))
        G = np.abs(data[max_idx] - mean) / std

        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        threshold = (n - 1) / np.sqrt(n) * np.sqrt((t_critical ** 2) / (n - 2 + (t_critical ** 2)))

        if G > threshold:
            data = np.delete(data, max_idx)
        else:
            is_outlier = False

    return data


def find_unique_closest(L1, L2):
    """
    Find the event in L1 that is closest to each event in L2. Each event in L1 can only be used once.
    :param L1:
    :param L2:
    :return:
    """
    result = []
    isUsed = [False] * len(L1)
    for t in L2:
        flag = False
        for i in range(len(L1) - 1, -1, -1):
            if isUsed[i] == True:
                break
            if isUsed[i] == False and L1[i] <= t:
                result.append(L1[i])
                isUsed[i] = True
                flag = True
                break
        if flag == False:
            result.append(-1)
    return result


def reactionTime(filePath):
    """
    Find the reaction time between the U-turn point and its previous event (pacman, energizer, ghost)
    :param filePath:
    :return:
    """
    print(filePath)
    data = pd.read_pickle(filePath)
    if "DayTrial" not in data.columns:
        data["DayTrial"] = copy.deepcopy(list(data["file"]))
    data["game"] = data.DayTrial.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )
    reactionTimes = []
    events = []
    for idx, grp in data.groupby("game"):
        energizers = list(grp["energizers"])
        energizers = [[] if isinstance(e, float) else e for e in energizers]
        energizers = [len(eval(e)) if isinstance(e, str) else len(e) for e in energizers]
        eatEIndex = [i for i, x in enumerate(energizers) if i != 0 and x != energizers[i - 1]]

        IS1 = np.array(grp["ifscared1"]).tolist()
        IS2 = np.array(grp["ifscared2"]).tolist()
        eatG1Index = [i for i, x in enumerate(IS1) if x == 3 and x != IS1[i - 1]]
        eatG2Index = [i for i, x in enumerate(IS2) if x == 3 and x != IS2[i - 1]]
        eatGIndex = list(set(eatG1Index + eatG2Index))

        eventIndexEG = list(set(eatEIndex + eatGIndex))
        eventIndexEG.sort()

        eventIndex = eventIndexEG
        # Find U-turn events
        pacmanDir = list(grp["pacman_dir"])
        DirToNum = {
            "left": 0, "right": 1, "up": 2, "down": 3
        }
        pacmanDir = [DirToNum[dir] if isinstance(dir, str) else -1 for dir in pacmanDir]
        for i in range(len(pacmanDir)):
            if pacmanDir[i] == -1 and i != 0:
                pacmanDir[i] = pacmanDir[i - 1]

        isTurnRound = {
            (0, 1): True, (1, 0): True, (2, 3): True, (3, 2): True
        }
        UTurnRoundIndex = [i for i in range(1, len(pacmanDir)) if
                           isTurnRound.__contains__((pacmanDir[i - 1], pacmanDir[i]))]

        # Find turn events
        turnRoundIndex = [i for i in range(1, len(pacmanDir)) if
                          not isTurnRound.__contains__((pacmanDir[i - 1], pacmanDir[i])) and pacmanDir[i - 1] !=
                          pacmanDir[i]]

        # Find the previous event of the U-turn event
        turnRoundEvent = find_unique_closest(eventIndex, UTurnRoundIndex)
        for i in range(len(turnRoundEvent)):
            if turnRoundEvent[i] == -1:
                continue
            # If there are  turning events between the U-turn event and the previous event
            temp = [1 if Turn > turnRoundEvent[i] and Turn < UTurnRoundIndex[i] else 0 for Turn in turnRoundIndex]
            if np.sum(temp) > 0:
                continue
            reactionTimes.append((UTurnRoundIndex[i] - turnRoundEvent[i]) * 25)
            if turnRoundEvent[i] in eatEIndex:
                events.append("E")
            elif turnRoundEvent[i] in eatGIndex:
                events.append("G")
    if len(reactionTimes) > 0:
        reactionTimes = list(grubbs_test(reactionTimes))
        meanReactionTime = np.mean(reactionTimes)
    else:
        meanReactionTime = []
        reactionTimes = []
        events = []

    return meanReactionTime, reactionTimes, events


def reaction_time_monkey(Monkey, date):
    fileFolderHuman = "../MonkeyData/FrameData/" + date + "/"
    fileNamesHuman = os.listdir(fileFolderHuman)
    filePathsHuman = [fileFolderHuman + fileName for fileName in fileNamesHuman if Monkey.lower() in fileName.lower()]

    # result = []
    # for i in range(2):
    #     result.append(reactionTime(filePathsHuman[0]))
    with multiprocessing.Pool(processes=12) as pool:
        result = pool.map(partial(reactionTime), filePathsHuman)

    reactionTimePerEvent = sum([r[1] for r in result if len(r) > 0], [])
    meanReactionTime = np.mean(reactionTimePerEvent)
    events = sum([r[2] for r in result if len(r) > 0], [])
    data = {
        "meanReactionTime": meanReactionTime,
        "reactionTimePerEvent": reactionTimePerEvent,
        "events": events,
    }
    pd.to_pickle(data, "../MonkeyData/Performance/" + date + "/" + "reactionTimeMonkey" + Monkey[0] + ".pkl")


def reward(filePath):
    beanReward = 2
    energizerReward = 4
    ghostReward = 10
    ghostPenalty = 5

    print(filePath)
    data = pd.read_pickle(filePath)
    if "DayTrial" not in data.columns:
        data["DayTrial"] = copy.deepcopy(list(data["file"]))
    data["game"] = data.DayTrial.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )

    rewardPerGame = []
    tilePerGame = []
    eatGhostNum = []
    for idx, grp in data.groupby("game"):
        try:
            energizersReward = (len(eval(grp["energizers"].iloc[0])) - len(
                eval(grp["energizers"].iloc[-1]))) * energizerReward
        except:
            energizers = list(grp["energizers"])
            energizers = [[] if isinstance(e, float) else e for e in energizers]
            energizersReward = (len(energizers[0]) - len(energizers[-1])) * energizerReward

        deadPenalty = (len(list(grp.groupby("DayTrial"))) - 1) * ghostPenalty

        IS1 = np.array(grp["ifscared1"]).tolist()
        IS2 = np.array(grp["ifscared2"]).tolist()
        IS1 = [x for i, x in enumerate(IS1) if i == 0 or x != IS1[i - 1]]
        IS2 = [x for i, x in enumerate(IS2) if i == 0 or x != IS2[i - 1]]
        numberEatGhost1 = np.where(np.array(IS1) == 3)[0].shape[0]
        numberEatGhost2 = np.where(np.array(IS2) == 3)[0].shape[0]
        eatGhostNum.append(numberEatGhost1 + numberEatGhost2)
        eatGhostRward = (numberEatGhost1 + numberEatGhost2) * ghostReward

        gameReward = energizersReward + eatGhostRward - deadPenalty
        rewardPerGame.append(gameReward)
        tilePerGame.append(len(grp))
    return rewardPerGame, tilePerGame, eatGhostNum


def reward_monkey(Monkey, date):
    fileFolderHuman = "../MonkeyData/CorrectedWeightData/" + date + "/"
    fileNamesHuman = os.listdir(fileFolderHuman)
    filePathsHuman = [fileFolderHuman + fileName for fileName in fileNamesHuman if Monkey.lower() in fileName.lower()]

    # result = []
    # for i in range(2):
    #     result.append(reward(filePathsHuman[i]))
    with multiprocessing.Pool(processes=12) as pool:
        result = pool.map(partial(reward), filePathsHuman)

    rewardPerGame = sum([r[0] for r in result], [])
    tilePerGame = sum([r[1] for r in result], [])
    eatGhostNum = sum([r[2] for r in result], [])
    result = {
        "rewardPerGame": rewardPerGame,
        "tilePerGame": tilePerGame,
        "eatGhostNum": eatGhostNum,
    }
    pd.to_pickle(result, "../MonkeyData/Performance/" + date + "/rewardMonkey" + Monkey[0] + ".pkl")


if __name__ == '__main__':
    Monkey = "Omega"
    date = "Year3"
    reaction_time_monkey(Monkey, date)
    reward_monkey(Monkey, date)
    Monkey = "Patamon"
    reaction_time_monkey(Monkey, date)
    reward_monkey(Monkey, date)
