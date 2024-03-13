import copy

import pandas as pd
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering


def DividePerson(date):
    results = []
    path = "../HumanData/Grammar/" + date + "/"
    filenames = os.listdir(path)
    for k, filename in enumerate(filenames):
        result = pd.read_pickle(path + filename)
        results.append(copy.deepcopy(result))

    allGrammarBook = []
    allComponents = []
    for k in range(len(results)):
        grammar = results[k]["sets"]
        allGrammarBook += grammar
        component = results[k]["components"]
        allComponents += component

    newAllGrammarBook = []
    newAllComponents = []
    for i in range(len(allGrammarBook)):
        if allGrammarBook[i] not in newAllGrammarBook:
            newAllGrammarBook.append(allGrammarBook[i])
            newAllComponents.append(allComponents[i])

    features = np.zeros((len(results), 3))
    for k in range(len(results)):
        grammar = copy.deepcopy(results[k]["sets"])
        frequency = results[k]["frequency"]
        if results[k]["skipGramNum"] != 0:
            grammar.append("skip")
            frequency = np.append(frequency, results[k]["skipGramNum"])
        pro = frequency / np.sum(frequency)
        for i, g in enumerate(grammar):
            index = len(g) - 1 if len(g) <= 3 else 2
            features[k][index] += pro[i]

    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    labels = model.fit_predict(features)

    index1 = np.where(labels == 0)[0]
    print(index1)

    index2 = np.where(labels == 1)[0]
    print(index2)

    grammarBook1 = []
    grammarBook2 = []
    componentBook1 = []
    componentBook2 = []
    for k in range(len(results)):
        grammar = results[k]["sets"] + ['N']
        component = results[k]["components"] + [['N', '']]
        if k in index1:
            grammarBook1 += grammar
            componentBook1 += component
        else:
            grammarBook2 += grammar
            componentBook2 += component

    newGrammarBook1 = []
    newComponentBook1 = []
    for i in range(len(grammarBook1)):
        if grammarBook1[i] not in newGrammarBook1:
            newGrammarBook1.append(grammarBook1[i])
            newComponentBook1.append(componentBook1[i])
    grammarBook1 = newGrammarBook1
    componentBook1 = newComponentBook1

    allGrammarBookIndex = sorted(enumerate(grammarBook1), key=lambda x: x[1])
    allGrammarBookIndex = [i[0] for i in allGrammarBookIndex]
    grammarBook1 = list(np.array(grammarBook1)[allGrammarBookIndex])
    componentBook1 = list(np.array(componentBook1)[allGrammarBookIndex])

    allGrammarBookIndex = sorted(enumerate(grammarBook1), key=lambda x: len(x[1]))
    allGrammarBookIndex = [i[0] for i in allGrammarBookIndex]
    grammarBook1 = list(np.array(grammarBook1)[allGrammarBookIndex])
    componentBook1 = list(np.array(componentBook1)[allGrammarBookIndex])

    newGrammarBook2 = []
    newComponentBook2 = []
    for i in range(len(grammarBook2)):
        if grammarBook2[i] not in newGrammarBook2:
            newGrammarBook2.append(grammarBook2[i])
            newComponentBook2.append(componentBook2[i])
    grammarBook2 = newGrammarBook2
    componentBook2 = newComponentBook2

    allGrammarBookIndex = sorted(enumerate(grammarBook2), key=lambda x: x[1])
    allGrammarBookIndex = [i[0] for i in allGrammarBookIndex]
    grammarBook2 = list(np.array(grammarBook2)[allGrammarBookIndex])
    componentBook2 = list(np.array(componentBook2)[allGrammarBookIndex])

    allGrammarBookIndex = sorted(enumerate(grammarBook2), key=lambda x: len(x[1]))
    allGrammarBookIndex = [i[0] for i in allGrammarBookIndex]
    grammarBook2 = list(np.array(grammarBook2)[allGrammarBookIndex])
    componentBook2 = list(np.array(componentBook2)[allGrammarBookIndex])

    print(grammarBook1)
    print([list(g) for g in componentBook1])
    print(grammarBook2)
    print([list(g) for g in componentBook2])
    #
    for k in range(len(results)):
        if k in index1:
            results[k]["sets"] = grammarBook1
            results[k]["components"] = componentBook1
        else:
            results[k]["sets"] = grammarBook2
            results[k]["components"] = componentBook2
        fileName = "../HumanData/GrammarCluster/" + date + "/" + results[k]["fileNames"][0]
        pd.to_pickle(results[k], fileName)


def GrammarAlign(Type, date=None):
    if Type == "Human":
        fileFolder = "../HumanData/GrammarCluster/" + date + "/"
        saveFolder = "../HumanData/GrammarFinall/" + date + "/"
    else:
        fileFolder = "../MonkeyData/Grammar/" + date + "/"
        saveFolder = "../MonkeyData/GrammarFinall/" + date + "/"
    fileNames = os.listdir(fileFolder)
    for i in range(len(fileNames)):
        result = pd.read_pickle(fileFolder + fileNames[i])
        sets = result["sets"]
        sequence = result["sequence"]
        P = [0] * len(sets)
        for k in range(len(sequence)):
            seq = sequence[k:]
            for j in range(len(sets)):
                l = len(sets[j])
                if len(seq) >= l and seq[:l] == sets[j]:
                    P[j] += 1
        P = np.array(P) / np.sum(P)
        result["P"] = P
        pd.to_pickle(result, saveFolder + fileNames[i])


if __name__ == '__main__':
    # DividePerson()
    date = "session2"
    DividePerson(date)
    GrammarAlign("Human", date)
    date = "Year3"
    GrammarAlign("Monkey", date)
