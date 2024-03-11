import copy
import itertools
import numpy as np
import pandas as pd
from utility import generate_all_Markov_network, gibbs_sampling
import os
import multiprocessing
from functools import partial
import pickle
import matplotlib.pyplot as plt
from PCalgorithm import PCskletetonData

node_number = 6
networks = generate_all_Markov_network(node_number)
# plt.rcParams["font.sans-serif"] = "CMU Serif"
plt.rcParams['font.family'] = 'CMU Serif'


# 马尔可夫网络抽样
def generate_simulation_data_par(num_sample, m):
    print(num_sample)
    edge_potentials = []
    node_datas = []
    graphs = []
    G = []
    for i in range(m):
        graph = np.random.randint(0, len(networks) - 1, size=1)[0]
        graph = networks[graph]
        data, edge_potential, learned_graph = gibbs_sampling(num_sample, graph)
        node_datas.append(data)
        edge_potentials.append(edge_potential)
        graphs.append(graph)
        G.append(learned_graph)
    data = {
        "graphs": graphs,
        "node_datas": node_datas,
        "edge_potentials": edge_potentials,
        "G": G
    }
    return data


def generate_simulation_data(m=1000):
    num_samples = list(range(100, 3001, 100))
    # generate_simulation_data_par(num_samples[0], m)
    with multiprocessing.Pool(processes=12) as pool:
        data = pool.map(partial(generate_simulation_data_par, m=m), num_samples)

    simData = {}
    for i in range(len(num_samples)):
        simData.update({num_samples[i]: data[i]})
    with open("./data/MNData.pkl", "wb") as file:
        pickle.dump(simData, file)


def MarkovNetworkAccuracy():
    # with open("./data/MNData.pkl", "rb") as file:
    #     datas = pickle.load(file)

    path = "./data/MarkovJust/"
    fileNames = os.listdir(path)
    filePaths = [path + filename for filename in fileNames]
    filePaths.sort()
    datas = {}
    for filepath in filePaths:
        print(filepath)
        d = pd.read_pickle(filepath)
        d["node_datas"] = 0
        name = int(filepath.split("/")[-1][6:][:-4])
        datas.update({name: d})
    accuracies = []
    stds = []
    for num_sample in list(datas.keys()):
        data = datas[num_sample]
        MarkovNetworks = data["G"]
        accuracy = 0
        tempAccuracy = []
        for i, g in enumerate(MarkovNetworks):
            original_graph = data["graphs"][i][0]
            # if np.sum(original_graph) == 0:
            #     continue
            temp = (g == original_graph)
            rows, cols = np.triu_indices(temp.shape[0], k=1)
            # Extract values
            temp = temp[rows, cols]
            # temp = temp[original_graph == 1]
            # index = np.where(original_graph == 1)
            accuracy += np.sum(temp)
            tempAccuracy.append(np.mean(temp))
        # accuracy /= (len(temp) * num)
        accuracies.append(np.mean(tempAccuracy))
        stds.append(np.std(tempAccuracy))
    return list(datas.keys()), np.array(accuracies), np.array(stds)


def BeliefNetworkAccuracy():
    # with open("./data/BNData.pkl", "rb") as file:
    #     datas = pickle.load(file)

    path = "./data/Bayesian/"
    fileNames = os.listdir(path)
    filePaths = [path + filename for filename in fileNames]
    filePaths.sort()
    datas = {}
    for filepath in filePaths:
        print(filepath)
        d = pd.read_pickle(filepath)
        d["node_datas"] = 0
        name = int(filepath.split("/")[-1][6:][:-4])
        datas.update({name: d})

    accuracies = []
    stds = []
    for num_sample in list(datas.keys()):
        data = datas[num_sample]
        MarkovNetworks = data["G"]
        accuracy = 0
        tempAccuracy = []
        for i, g in enumerate(MarkovNetworks):
            if g is None:
                continue
            original_graph = data["graphs"][i]
            temp = (g == original_graph)

            temp = temp[:, -2:]
            # temp = temp[original_graph == 1]
            # index = np.where(original_graph == 1)
            accuracy += np.sum(temp)
            tempAccuracy.append(np.mean(temp))
        accuracy /= (15 * len(MarkovNetworks))
        accuracies.append(np.mean(tempAccuracy))
        stds.append(np.std(tempAccuracy))
    return list(datas.keys()), np.array(accuracies), np.array(stds)


if __name__ == '__main__':
    # generate_simulation_data(m=1000)
    # getMarkovNetwork()
    # index, BayesianNetworkAccuracy, BayesianMarkovNetworkStd = BeliefNetworkAccuracy()
    # MarkovNetworkStd = BayesianMarkovNetworkStd
    # MarkovNetworkAccuracy = BayesianNetworkAccuracy
    index, MarkovNetworkAccuracy, MarkovNetworkStd = MarkovNetworkAccuracy()
    fig = plt.figure(figsize=(16, 16), dpi=300)
    font_properties = {'size': 40}
    plt.plot(index, MarkovNetworkAccuracy, color='blue')

    plt.fill_between(index, MarkovNetworkAccuracy - MarkovNetworkStd, MarkovNetworkAccuracy + MarkovNetworkStd,
                     color='blue', alpha=0.1)

    # plt.plot(index, BayesianNetworkAccuracy, color='orange')
    #
    # plt.fill_between(index, BayesianNetworkAccuracy - BayesianMarkovNetworkStd,
    #                  BayesianNetworkAccuracy + BayesianMarkovNetworkStd,
    #                  color='orange', alpha=0.1)
    #
    #
    # plt.legend(["Markov Network", "Belief Network"], prop=font_properties)
    plt.xlabel("Sample number", fontdict=font_properties)
    plt.ylabel("Accuracy", fontdict=font_properties)
    plt.yticks(fontsize=font_properties["size"],
               )
    plt.xticks(fontsize=font_properties["size"],
               )
    plt.tight_layout()
    # plt.savefig("SupFig-sim1-PGM-M-Just.pdf")
    plt.show()
    # data = pd.read_pickle("./data/LMNData.pkl")
    # getMarkovNetwork()
    # generate_simulation_data(m=1000)
