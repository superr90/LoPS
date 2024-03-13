'''
Description:
    Convert zero Q values to -inf.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
'''
import os
import pickle
import sys

current_directory = os.path.dirname(os.path.abspath("./UtilityConvert.py"))
root_path = os.path.abspath(os.path.dirname(os.path.dirname(current_directory)) + os.path.sep + ".")
sys.path.append(root_path)
import numpy as np
import pandas as pd
import copy
from Utils.FileUtils import readAdjacentMap
import multiprocessing
from functools import partial


def _infConvert(cur_pos, cur_Q, adjacent_data):
    new_Q = copy.deepcopy(cur_Q)
    adjacent_pos = adjacent_data[cur_pos]
    for dir_idx, dir in enumerate(["left", "right", "up", "down"]):
        if adjacent_pos[dir] is None or isinstance(adjacent_pos[dir], float):
            new_Q[dir_idx] = -np.inf
    return new_Q


def infUtilityConvert(filename, adjacent_data):
    data = pd.read_pickle(filename)
    # convert to inf
    for a in ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer", "no_energizer"]:
        data["{}_Q".format(a)] = data[["pacmanPos", "{}_Q".format(a)]].apply(
            lambda x: _infConvert(x.pacmanPos, x["{}_Q".format(a)], adjacent_data), axis=1
        )
    return data


def CorrectUtility(filename, adjacent_data, saveFolder):
    data = infUtilityConvert(filename, adjacent_data)
    savePath = saveFolder + filename.split("/")[-1]
    print(filename)
    data.to_pickle(savePath)


def correct_monkey_utility(date):
    if date == "Year1":
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map_before.csv")
    else:
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map.csv")

    fileFolder = "../MonkeyData/UtilityData/" + date + "/"
    saveFolder = "../MonkeyData/CorrectedUtilityData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + f for f in fileNames]

    with multiprocessing.Pool(processes=12) as pool:
        pool.map(
            partial(CorrectUtility, adjacent_data=adjacent_data, saveFolder=saveFolder), filePaths)


if __name__ == '__main__':
    date = "Year3"
    correct_monkey_utility(date)
