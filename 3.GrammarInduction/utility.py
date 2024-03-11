import pandas as pd
import os
from copy import deepcopy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

K = 34

def nearestNeighbors(fileFolder):
    """
    根据特征找到每个人最近的n个邻居
    :param Type: 
    :return: 
    """

    keys = ['global', 'local', 'evade_blinky', 'evade_clyde', 'approach', 'energizer', 'no_energizer', "stay"]
    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + fileName for fileName in fileNames]

    # 获取特征
    features = []
    for i, filePath in enumerate(filePaths):
        data = pd.read_pickle(filePath)
        feature = (data[keys] - 1).values
        feature = np.sum(feature, axis=0)
        feature = feature / np.sum(feature)
        features.append(deepcopy(feature))

    features = np.array(features)
    data_min = features.min(axis=0)
    data_max = features.max(axis=0)
    features = (features - data_min) / (data_max - data_min)
    features[:, [4, 5, 7]] = 2 * features[:, [4, 5, 7]]

    nn = NearestNeighbors(n_neighbors=len(features))
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    neighbors = []
    for i in range(len(distances)):
        dis = distances[i, :].reshape(-1, 1)
        gate = 1
        while True:
            temp = np.where(dis <= gate)[0]
            if len(temp) < 5:
                gate += 0.1
                continue
            neighbor = list(indices[i][temp])
            neighbors.append(neighbor)
            print(i, neighbor)
            break
    return neighbors, fileNames


def getFeature(Type="ghost2"):
    """
    根据特征找到每个人最近的n个邻居
    :param Type: 
    :return: 
    """

    if Type == "ghost2":
        keys = ['global', 'local', 'evade_blinky', 'evade_clyde', 'approach', 'energizer', 'no_energizer', "stay"]
        fileFolder = "../../MyData/fmriFormedData2/"
    else:
        keys = ['global', 'local', 'evade', 'approach', 'energizer',
                'no_energizer', "stay"]
        fileFolder = "../../MyData/fmriFormedData4/"

    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + fileName for fileName in fileNames]

    # 获取特征
    features = []
    for i, filePath in enumerate(filePaths):
        data = pd.read_pickle(filePath)
        feature = (data[keys] - 1).values
        feature = np.sum(feature, axis=0)
        feature = feature / np.sum(feature)
        features.append(deepcopy(feature))

    features = np.array(features)
    return features


def toOnehot(x, agent_num):
    strategy = np.ones(agent_num, dtype=np.int64)
    strategy[x] = 2
    return strategy
