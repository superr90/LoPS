# Encoding=utf8
import pandas as pd
import pickle as pickle
import numpy as np
import copy
import math
from PGM.Utils import *
import os
from functools import partial
import multiprocessing
import warnings

warnings.filterwarnings("ignore")


def screenFirstPoiontPerStrategy(data, strategyName):
    """
    For each continuous strategy, select the first strategy point and delete other points.
    :param data: dataframe
    :return:
    """
    data["game"] = data.file.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )
    news_datas = pd.DataFrame()
    gameDemarc = []
    for idx, grp in data.groupby("game"):
        first_point_mask = (grp[strategyName] == grp[strategyName].shift(1)).sum(1) < len(
            strategyName
        )
        news_datas = news_datas.append(grp.loc[first_point_mask])
        gameDemarc += [0] * (len(grp.loc[first_point_mask]) - 1) + [1]
    news_datas = news_datas.sort_index()
    news_datas.reset_index(drop=True, inplace=True)
    news_datas.drop(columns="game", inplace=True)
    news_datas.reset_index(drop=True, inplace=True)
    news_datas["gameDemarc"] = gameDemarc

    return news_datas


def combineEvade(data, status_cols):
    evade = []
    for i in range(len(data)):
        index = np.array(data[status_cols].iloc[i])
        index = np.argmax(index)
        if index in [2, 3]:
            evade.append(2)
        else:
            evade.append(1)

    data["evade"] = evade
    return data


def strategyToOnehot(x, agent_num):
    strategy = np.ones(agent_num, dtype=np.int)
    strategy[x] = 2
    return strategy


def getOnehotStrategy(data):
    """
    Convert the strategy column into onehot form. Each column represents whether a strategy is enabled or not.
    :param data:
    :return:
    """

    data.reset_index(drop=True, inplace=True)
    statusCols = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer",
                  "no_energizer", "vague", "stay"]
    status = np.stack(data["strategy"].apply(lambda x: strategyToOnehot(x, len(statusCols))))
    statusEncode = (
            pd.DataFrame(status == status.max(1, keepdims=True), columns=statusCols).astype(
                int
            )
            + 1
    )

    # Adjust column position
    columns = list(data.columns)
    data[statusCols] = statusEncode
    columns = columns[:-2] + statusCols + columns[-2:]
    data = data[columns]

    data = combineEvade(data, statusCols)
    status_cols = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer", "no_energizer",
                   "vague", "stay"]
    return data, status_cols


def dataChange(path):
    """
    Preliminary conversion of data: 1. Discretization 2. Production strategy 3. Continuous strategy to select the first point
    :param filename:
    :param save_path:
    :param need_discretize:
    :return:
    """
    fileName = path[0]
    savePath = path[1]
    print(fileName)
    try:
        data = pd.read_pickle(fileName)
    except:
        with open(fileName) as f:
            data = pickle.load(f)
    data, strategyName = getOnehotStrategy(data)
    data = screenFirstPoiontPerStrategy(data, strategyName)
    data.to_pickle(savePath)


def changeMonkeyDataEveryMonkey(monkey, date):
    """
    Consolidate people's data together
    :return:
    """
    filefolder = "../MonkeyData/FormedData/" + date + "/"
    states_keys = ['IS1', 'IS2', 'PE', 'PG1', 'PG2', 'BN5', 'BN10']
    strategy_keys = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer", "no_energizer", "stay",
                     "vague"]
    if monkey == "omega":
        savepath = "../MonkeyData/seq/" + date + "/" + "Omega.pkl"
    else:
        savepath = "../MonkeyData/seq/" + date + "/" + "Patamon.pkl"

    filenames = os.listdir(filefolder)
    filenames = [filename for filename in filenames if monkey.lower() in filename.lower()]

    data = []
    points = [-1]
    for filename in filenames:
        temp_data = pd.read_pickle(filefolder + filename)
        points.append(points[-1] + len(temp_data))
        data.append(copy.deepcopy(temp_data))
    data = pd.concat(data, axis=0)

    states = data[states_keys]
    strategy = data[strategy_keys]

    S = ["G", "L", '1', '2', "A", "E", "N", "S", "V"]
    seq = ""
    seqs = []
    t = ""
    points = points[1:]
    for i in range(len(data)):
        temp = data[['global', 'local', "evade_blinky", "evade_clyde", 'approach', 'energizer', "no_energizer", "stay",
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

    }
    with open(savepath, "wb") as file:
        pickle.dump(result, file)


def form_date_monkey(date):
    fileFolder = "../MonkeyData/DiscreteFeatureData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    paths = []
    for filename in fileNames:
        print(filename)
        savepath = "../MonkeyData/FrameData/" + date + "/" + filename
        paths.append((fileFolder + filename, savepath))

    with multiprocessing.Pool(processes=12) as pool:
        pool.map(
            partial(dataChange, need_discretize=False), paths)

    changeMonkeyDataEveryMonkey("omega", date)
    changeMonkeyDataEveryMonkey("Patamon", date)


if __name__ == '__main__':
    date = "Year3"
    form_date_monkey(date)
