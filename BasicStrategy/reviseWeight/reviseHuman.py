import pickle
import numpy as np
import pandas as pd
import os
import copy
from functools import partial
import multiprocessing
from itertools import groupby
from operator import itemgetter
import warnings
from utility import revise_function, strategyFromWeight, revise_vague, reviseWrongEnergizer, reviseMain

warnings.filterwarnings("ignore")


def correct_human_weight(date):
    scared_time = 63
    strategy_number = {
        "global": 0, "local": 1, "evade_blinky": 2, "evade_clyde": 3, "evade_3": 4, "evade_4": 5, "approach": 6,
        "energizer": 7, "no_energizer": 8,
        "vague": 9, "stay": 10
    }
    agents = ["global", "local", "evade_blinky", "evade_clyde", "evade_ghost3", "evade_ghost4", "approach",
              "energizer", "no_energizer"]
    agent_num = len(agents)
    suffix = "_Q_norm"
    agents_list = [("{}" + suffix).format(each) for each in agents]
    filefolder = "../../HumanData/WeightData/" + date + "/"
    saveFolder = "../../HumanData/CorrectedWeightData/" + date + "/"
    filenames = os.listdir(filefolder)
    filenames = [filefolder + filename for filename in filenames]
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(
            partial(reviseMain, savePath=saveFolder, strategy_number=strategy_number, agents=agents,
                    agents_list=agents_list, agent_num=agent_num, scared_time=scared_time, suffix=suffix), filenames)


if __name__ == '__main__':
    date = "session2"
    correct_human_weight(date)
