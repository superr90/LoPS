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


# Markov network sampling
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

    path = "./data/Markov/"
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


def JustBeliefNetworkAccuracy():
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
    # index, BayesianNetworkAccuracy, BayesianMarkovNetworkStd = BeliefNetworkAccuracy()
    # index, MarkovNetworkAccuracy, MarkovNetworkStd = MarkovNetworkAccuracy()
    data = pd.read_pickle("./data/accuracy.pkl")
    index = data['index']

    BA = data['BA']
    BS = data['BS']
    JBA = data['JBA']
    JBS = data['JBS']
    MA = data['MA']
    MS = data['MS']

    colors = ["#B4585F", "#80B6B1", "#F6D087"]

    fig = plt.figure(figsize=(16, 16), dpi=300)
    font_properties = {'size': 40}
    plt.plot(index, MA, color=colors[0], label='Markov network')
    plt.fill_between(index, MA - MS, MA + MS,
                     color=colors[0], alpha=0.1)

    plt.plot(index, JBA, color=colors[1], label='Bayesian network')
    plt.fill_between(index, JBA - JBS, JBA + JBS,
                     color=colors[1], alpha=0.1)

    plt.plot(index, BA, color=colors[2], label='Hybrid network')
    plt.fill_between(index, BA - BS, BA + BS,
                     color=colors[2], alpha=0.1)

    plt.legend(prop=font_properties)
    plt.xlabel("Sample Size", fontdict=font_properties)
    plt.ylabel("Accuracy", fontdict=font_properties)
    plt.yticks(fontsize=font_properties["size"],
               )
    plt.xticks(fontsize=font_properties["size"],
               )
    plt.tight_layout()
    # plt.savefig("SupFig-sim-PGM.pdf")
    plt.show()
