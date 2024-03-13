import pickle

import numpy as np
import pandas as pd
import os
import copy

from functools import partial
import multiprocessing

import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")
from itertools import groupby
from operator import itemgetter
from utility import _makeChoice, _calculate_is_correct, _oneHot
from utility import revise_function, strategyFromWeight, revise_vague, reviseWrongEnergizer, reviseMain


def correct_monkey_weight(date):
    scared_time = 28
    strategy_number = {
        "global": 0, "local": 1, "evade_blinky": 2, "evade_clyde": 3, "approach": 4, "energizer": 5, "no_energizer": 6,
        "vague": 7, "stay": 8
    }
    suffix = "_Q_norm"
    agents = ["global", "local", "evade_blinky", "evade_clyde", "approach",
              "energizer", "no_energizer"]
    agent_num = len(agents)
    agents_list = [("{}" + suffix).format(each) for each in agents]

    filefolder = "../../MonkeyData/WeightData/" + date + "/"
    filenames = os.listdir(filefolder)
    filenames = [filefolder + filename for filename in filenames]
    saveFolder = "../../MonkeyData/CorrectedWeightData/" + date + "/"
    with multiprocessing.Pool(processes=12) as pool:
        pool.map(
            partial(reviseMain, savePath=saveFolder, strategy_number=strategy_number, agents=agents,
                    agents_list=agents_list, agent_num=agent_num, scared_time=scared_time, suffix=suffix), filenames)


if __name__ == '__main__':
    date = "Year3"
    correct_monkey_weight(date)
