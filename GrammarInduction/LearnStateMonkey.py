import os
import pandas as pd
from PGM.PCalgorithm import *
import pandas as pd
import pickle
import numpy as np
import random


def PC(data):
    data_num = data.shape[1]
    index = np.random.choice(a=list(range(data_num)), p=[1 / data_num] * data_num, replace=False, size=4000)
    data = data[:, index]
    G, S = PCskletetonData(data)
    return G


def learm_state_graph_monkey(monkey, date):
    keys = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]
    result = pd.read_pickle("../MonkeyData/seq/" + date + "/" + monkey + ".pkl")
    save_path = "../MonkeyData/state/" + date + "/" + monkey + ".pkl"
    states = result["state"][keys]
    stateNames = list(states.columns)
    choice = np.random.choice(a=list(range(len(states))), p=[1 / len(states) for i in range(len(states))],
                              size=len(states),
                              replace=False)
    data = states.iloc[choice][stateNames].values.T
    data = data + 1
    G = PC(data)
    result = {
        "G": G,
        "stateNames": stateNames,
        "data": states.values.T + 1
    }
    with open(save_path, "wb") as file:
        pickle.dump(result, file)



if __name__ == '__main__':
    date = "Year3"
    monkey = "Omega"
    learm_state_graph_monkey(monkey, date)
    monkey = "Patamon"
    learm_state_graph_monkey(monkey, date)
