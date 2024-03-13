import os
from PGM.PCalgorithm import *
import pandas as pd
import pickle
import numpy as np

def PC(data, sampleNumber=4000):
    data_num = data.shape[1]
    if sampleNumber > data_num:
        sampleNumber = data_num
    index = np.random.choice(a=list(range(data_num)), p=[1 / data_num] * data_num, replace=False, size=sampleNumber)
    data = data[:, index]
    G, S = PCskletetonData(data)
    return G

def learm_state_graph_human(date):
    """
    Get the relationship between states
    :return:
    """
    fileFolder = "../HumanData/seq/" + date + "/"
    savePath = "../HumanData/state/" + date + "/"
    stateNames = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]
    fileNames = os.listdir(fileFolder)
    for i, fileName in enumerate(fileNames):
        print(fileName)
        result = pd.read_pickle(fileFolder + fileName)
        states = result["state"]
        data = states[stateNames].values.T + 1
        G = PC(data, data.shape[1])
        result = {
            "G": G,
            "stateNames": stateNames,
            "data": data
        }
        with open(savePath + fileName, "wb") as file:
            pickle.dump(result, file)


if __name__ == '__main__':
    date = "session2"
    learm_state_graph_human(date)
