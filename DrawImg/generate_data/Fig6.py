import os
import pickle
from collections import Counter
from copy import deepcopy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from copy import deepcopy
from PGM.bayesianScore import BDscore
from d3blocks import D3Blocks
from pycirclize import Circos
from matplotlib.patches import Patch

plt.style.use('ggplot')
from pycirclize.parser import Matrix

stateNameChange = {
    "IS1": "$m_1$",
    "IS2": "$m_2$",
    "PG1": "$g_1$",
    "PG2": "$g_2$",
    "PE": "$e$",
    "BN5": "$bl$",
    "BN10": "$bf$",
    "PA0": "$tu$",
    "PA3": "$gh$"
}
strategyChange = {
    "global": "G",
    "local": "L",
    "evade": "e",
    "approach": "A",
    "no_energizer": "N",
    "stay": "S",
    "energizer": "E",
    "evade_blinky": "1",
    "evade_clyde": "2",
}


def getState(x, index):
    pass


def dividSubGraph(G, Type="Expert"):
    subGraphIndexs = []
    if Type == "Expert":
        subGraphIndexs = [np.array([0, 1, 2, 3, 4]), np.array([4, 5])]
        return subGraphIndexs
    GN = np.linalg.matrix_power(G, len(G))
    while len(np.where(GN != 0)[0]) != 0:
        index = list(set(list(np.where(GN != 0)[0])))[0]
        index = np.where(GN[index, :] != 0)[0]

        for i in index:
            GN[i, :] = [0] * len(GN[i, :])
            GN[:, i] = [0] * len(GN[:, i])
        subGraphIndexs.append(deepcopy(index))
    return subGraphIndexs


def filter(data, stateNumber, ):
    number = deepcopy(data[0])
    for i in range(1, len(data)):
        x = np.prod(stateNumber[:i])
        number += deepcopy(data[i]) * x

    counter = Counter(number)
    frequencies = {key: count / len(number) for key, count in counter.items()}
    keys = list(frequencies.keys())

    saveKeys = keys
    sample = []
    sampleIndex = []
    for i in range(len(number)):
        if number[i] not in sample:
            sample.append(number[i])
            sampleIndex.append(i)
    return number, saveKeys, sampleIndex


def stateOnehot(state):
    stateList = list(set(list(state)))

    columns = [str(s) for s in stateList]

    data = []
    for i in range(len(columns)):
        tempData = state == stateList[i]
        data.append(deepcopy(tempData))
    data = np.array(data, dtype=np.int) + 1

    return data, columns


def transitionData(resultData):
    strategy = resultData["strategy"]
    state = resultData["subGraphState"]
    saveState = resultData["subGraphSaveState"]

    s, sColumns = stateOnehot(state)
    a = strategy.values.T
    sa = np.concatenate((s, a), axis=0)
    sa_ = sa[:, :-1]
    s_ = s[:, 1:]
    transition = np.concatenate((sa_, s_), axis=0)
    columns = sColumns + list(strategy.columns) + [feature + '_' for feature in sColumns]
    transition = pd.DataFrame(transition.T)
    transition.columns = columns

    return transition, sColumns


def selectData(datas, strategy, dataType):
    if dataType == "Expert":
        removeStrategy = ["V"]
        strategyList = ["G", "L", "e", "A", "E", "N", "S"]
    elif dataType == "Amateur":
        removeStrategy = ["S", "V"]
        strategyList = ["G", "L", "e", "A", "E", "N"]
        removeStrategy = ["V"]
        strategyList = ["G", "L", "e", "A", "E", "N", "S"]

    else:
        removeStrategy = ["S", "V", "N"]
        strategyList = ["G", "L", "e", "A", "E"]
        removeStrategy = ["V"]
        strategyList = ["G", "L", "e", "A", "E", "N", "S"]

    selectedIndex = [j for j in range(len(strategy)) if strategy[j] not in removeStrategy]
    strategy = "".join(list(np.array(list(strategy))[selectedIndex]))
    datas = datas[:, selectedIndex]
    return datas, strategy, strategyList


def static():
    path = "../../HumanData/state_cluster/"
    stateFileNames = os.listdir(path)
    statePaths = [path + fileName for fileName in stateFileNames] + ["../../data/states monkey/monkey.pkl"]
    savePath = "../../HumanData/transition/transitionMatrix/"
    stateNumbers = np.array([3, 3, 2, 2, 2, 2, 2])

    path = "../../HumanData/seq_cluster/"
    seqFileNames = os.listdir(path)
    seqPaths = [path + fileName for fileName in seqFileNames] + ["../../data/seq monkey/monkey_seq.pkl"]

    dataTypes = ["Expert", "Amateur", "Monkey"]
    for i, path in enumerate(statePaths):
        strategy = pd.read_pickle(seqPaths[i])["seq"]
        strategy = strategy.replace("1", "e")
        strategy = strategy.replace("2", "e")
        selectedIndex = [j for j in range(len(strategy)) if j == 0 or strategy[j] != strategy[j - 1]]
        strategy = "".join(list(np.array(list(strategy))[selectedIndex]))

        stateGraph = pd.read_pickle(path)

        featureNames = stateGraph["stateNames"]
        datas = stateGraph["data"]
        datas = datas[:, selectedIndex]
        # datas, strategy, strategyList = selectData(datas, strategy, dataTypes[i])
        strategyList = ["G", "L", "e", "A", "E", "N", "S"]
        datas = pd.DataFrame(datas.T) - 1
        datas.columns = featureNames
        # datas["strategy"]=list(strategy)
        # temp1 = datas.values.T[:, :-1]
        # temp2 = np.array(list(strategy)).reshape(1, -1)[:, :-1]
        # temp3 = deepcopy(datas.values.T[:, 1:])
        # temp = np.vstack((temp1, temp2))
        # temp = np.vstack((temp, temp3))
        # index = np.where(temp[6,:] == 'A')[0]
        # temp_ = temp[:, index]

        G = stateGraph["G"]
        G = G + np.eye(len(G))

        subGraphIndexs = dividSubGraph(G, dataTypes[i])

        results = []
        for j, subGraphIndex in enumerate(subGraphIndexs):
            stateNumberOfFeature = list(np.array(stateNumbers)[subGraphIndex])
            featureName = list(np.array(featureNames)[subGraphIndex])
            tempData = datas[featureName].values.T

            number, saveKeys, sampleIndex = filter(tempData, stateNumberOfFeature)
            featureVector = tempData[:, sampleIndex]
            featureVector = pd.DataFrame(featureVector.T)
            featureVector.columns = featureName
            featureVector.index = saveKeys
            featureVector["state"] = saveKeys

            number_ = number[1:]
            number = number[:-1]
            # The dimension order is a s s'
            transitionMatrix = np.zeros((len(set(strategy)), len(saveKeys), len(saveKeys)))
            tempCount = [(strategy[k], number[k], number_[k]) for k in range(len(number))]
            counter = Counter(tempCount)
            keys = list(counter.keys())
            for k in range(len(keys)):
                key = keys[k]
                if key == ('G', 0, 18):
                    x = 0
                si = saveKeys.index(key[1])
                sj = saveKeys.index(key[2])
                if key[0] not in strategyList:
                    continue
                a = strategyList.index(key[0])
                transitionMatrix[a, si, sj] = counter[key]

            PS = np.sum(np.sum(transitionMatrix, axis=2), axis=0)
            PS = PS / np.sum(PS)
            thresholdPS = 1 / np.prod(PS.shape)
            # thresholdPS = 0
            saveStates = np.where(PS > thresholdPS)[0]
            transitionMatrix = transitionMatrix[:, saveStates, :][:, :, saveStates]

            # P(s'|s,a)
            CPS_ = np.zeros(transitionMatrix.shape)
            for k in range(transitionMatrix.shape[0]):
                for l in range(transitionMatrix.shape[1]):
                    temp = np.sum(transitionMatrix[k, l, :])
                    for m in range(transitionMatrix.shape[2]):
                        if temp != 0:
                            CPS_[k, l, m] = transitionMatrix[k, l, m] / temp

            # P(a|s,s')
            CPA = np.zeros((len(set(strategy)), len(saveKeys), len(saveKeys)))
            for l in range(transitionMatrix.shape[1]):
                for m in range(transitionMatrix.shape[2]):
                    temp = np.sum(transitionMatrix[:, l, m])
                    for k in range(transitionMatrix.shape[0]):
                        if temp != 0:
                            CPA[k, l, m] = transitionMatrix[k, l, m] / temp
            threshold1 = 1 / CPS_.shape[2]
            threshold2 = 0.002
            # threshold2 = 1 / np.prod(transitionMatrix.shape)
            resultMatrix = []
            for l in range(transitionMatrix.shape[1]):
                result = []
                for m in range(transitionMatrix.shape[2]):
                    s_ = ''
                    maxIndex = -1
                    for k in range(transitionMatrix.shape[0]):
                        if CPS_[k, l, m] > threshold1 and transitionMatrix[k, l, m] / np.sum(
                                transitionMatrix) > threshold2 and l != m:
                            s_ += strategyList[k]
                            if maxIndex == -1 or CPA[k, l, m] > CPA[maxIndex, l, m]:
                                maxIndex = k
                    if maxIndex != -1:
                        s = strategyList[maxIndex]
                    else:
                        s = ""
                    result.append(s_)
                resultMatrix.append(result)
            resultMatrix = np.array(resultMatrix)
            temp = [0] * len(strategyList)
            for l in range(resultMatrix.shape[0]):
                for m in range(resultMatrix.shape[1]):
                    for s in resultMatrix[l, m]:
                        k = strategyList.index(s)
                        temp[k] += 1
            x = 0

            # Characteristics corresponding to each state

            result = {
                "resultMatrix": resultMatrix,
                "featureVector": featureVector.iloc[saveStates],
                "PS": PS[saveStates],
                "state": list(np.array(saveKeys)[saveStates]),
                "stateNumberOfFeature": stateNumberOfFeature
            }
            results.append(deepcopy(result))

        with open(savePath + dataTypes[i] + ".pkl", 'wb') as file:
            pickle.dump(results, file)


def draw_node_image(ax, position, state, stateName):
    size = 0.1
    intervalSize = 0.05
    colors = []
    for i in range(len(stateName)):
        if stateName[i] == "IS1" or stateName[i] == "IS2":
            if state[i] == 0:
                colors.append("white")
            elif state[i] == 1:
                colors.append("gray")
            else:
                colors.append("black")
        else:
            if state[i] == 0:
                colors.append("white")
            else:
                colors.append("black")

    num1 = len(stateName) / 2
    num2 = num1 - 0.5
    for i, color in enumerate(colors):
        # Draw four squares with horizontal directions
        ax.add_patch(patches.Rectangle(
            (position[0] - num1 * size + i * size - num2 * intervalSize + i * intervalSize, position[1] - size / 2),
            size, size, facecolor=color))


def PlotGraph2D(transitionMatrix, sColumns, strategyNames, stateData, stateNumber, ax):
    stateName = list(stateData.columns)[:-1]
    stateData = stateData[stateName]
    matrixDict = {}
    G = nx.MultiDiGraph()
    edgesLabel = {}

    for i in range(np.prod(stateNumber)):
        G.add_node(i)
    for i in range(len(sColumns)):
        for j in range(len(sColumns)):
            tempValue = transitionMatrix[:, i, j]
            temp = np.where(tempValue == True)[0]
            if len(temp) != 0 and i != j:
                key = str(sColumns[i]) + "-" + str(sColumns[j])
                temp = list(np.array(strategyNames)[temp])
                matrixDict.update({key: temp})
                G.add_node(sColumns[i])
                G.add_node(sColumns[j])
                G.add_edge(sColumns[i], sColumns[j])
                edgesLabel.update({(sColumns[i], sColumns[j]): "".join(temp)})
    pos = nx.kamada_kawai_layout(G, scale=2)

    if len(G) == 0:
        return
        # Adjust position coordinates
    for key in pos:
        pos[key] = [3 * x for x in pos[key]]

    # Calculate corrected positions to ensure that the edge's start and end points are in the middle of the four squares
    mid_offsets = {node: (pos[node][0], pos[node][1]) for node in G.nodes}
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, connectionstyle="arc3,rad=0.2", ax=ax)
    for source, target, weight in G.edges:
        pos1 = pos[source]
        pos2 = pos[target]
        midPos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        offset = 0.2
        if pos1[0] <= pos2[0] and pos1[1] <= pos2[1]:
            h = -offset
            v = offset
        elif pos1[0] >= pos2[0] and pos1[1] <= pos2[1]:
            h = offset
            v = offset
        elif pos1[0] >= pos2[0] and pos1[1] >= pos2[1]:
            h = offset
            v = -offset
        elif pos1[0] <= pos2[0] and pos1[1] >= pos2[1]:
            h = -offset
            v = -offset
        midPos = (midPos[0] + v, midPos[1] + h)
        plt.text(x=midPos[0], y=midPos[1], s=edgesLabel[(source, target)])
    for node, position in mid_offsets.items():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            continue
        draw_node_image(ax, position, list(stateData.loc[node]), stateName)

    # ax.set_xlim(min(pos.values(), key=lambda x: x[0])[0] - 0.5, max(pos.values(), key=lambda x: x[0])[0] + 0.5)
    # ax.set_ylim(min(pos.values(), key=lambda x: x[1])[1] - 0.5, max(pos.values(), key=lambda x: x[1])[1] + 0.5)
    ax.text(0, 1, " ".join([stateNameChange[s] for s in stateName]), ha='left', va='top', transform=ax.transAxes)


def PlotGraph2D_(pos, G, featureVector, ax):
    featureName = list(featureVector.columns)[:-1]
    featureVector = featureVector[featureName]
    if len(G) == 0:
        return
    # Calculate corrected positions to ensure that the edge's start and end points are in the middle of the four squares
    mid_offsets = {node: (pos[node][0], pos[node][1]) for node in G.nodes}
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, connectionstyle="arc3,rad=0.2", ax=ax)
    for source, target, weight in G.edges:
        pos1 = pos[source]
        pos2 = pos[target]
        midPos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        offset = 0.2
        if pos1[0] <= pos2[0] and pos1[1] <= pos2[1]:
            h = -offset
            v = offset
        elif pos1[0] >= pos2[0] and pos1[1] <= pos2[1]:
            h = offset
            v = offset
        elif pos1[0] >= pos2[0] and pos1[1] >= pos2[1]:
            h = offset
            v = -offset
        elif pos1[0] <= pos2[0] and pos1[1] >= pos2[1]:
            h = -offset
            v = -offset
        midPos = (midPos[0] + v, midPos[1] + h)
        plt.text(x=midPos[0], y=midPos[1], s=G[source][target][0]["label"])

    # for node, position in mid_offsets.items():
    # if G.in_degree(node) == 0 and G.out_degree(node) == 0:
    #     continue
    # draw_node_image(ax, position, list(featureVector.loc[node]), featureName)

    min_x = min(pos.values(), key=lambda x: x[0])[0] - 0.5
    max_x = max(pos.values(), key=lambda x: x[0])[0] + 0.5
    min_y = min(pos.values(), key=lambda x: x[1])[1] - 0.5
    max_y = max(pos.values(), key=lambda x: x[1])[1] + 0.5
    print(min(pos.values(), key=lambda x: x[0])[0] - 0.5, max(pos.values(), key=lambda x: x[0])[0] + 0.5)
    print(min(pos.values(), key=lambda x: x[1])[1] - 0.5, max(pos.values(), key=lambda x: x[1])[1] + 0.5)
    ax.set_xlim(min(pos.values(), key=lambda x: x[0])[0] - 0.5, max(pos.values(), key=lambda x: x[0])[0] + 0.5)
    ax.set_ylim(min(pos.values(), key=lambda x: x[1])[1] - 0.5, max(pos.values(), key=lambda x: x[1])[1] + 0.5)
    ax.text(0, 1, " ".join([stateNameChange[s] for s in featureName]), ha='left', va='top', transform=ax.transAxes)
    # ax.set_xticks([min_x, max_x])
    # ax.set_yticks([min_y, max_y])
    # ax.axhline(y=0, color='k', linewidth=0.5)
    # ax.axvline(x=0, color='k', linewidth=0.5)


def PlotMain():
    dataTypes = ["Expert", "Amateur", "Monkey"]
    plt.figure(figsize=(20, 20))
    positions = []
    Gs = []
    for k, dataType in enumerate(dataTypes):
        positions.append([])
        Gs.append([])
        path = "../../MyData/transition/transitionMatrix/" + dataType + ".pkl"
        datas = pd.read_pickle(path)
        for l, data in enumerate(datas):
            state = data["state"]
            resultMatrix = data["resultMatrix"]
            featureVector = data["featureVector"]
            PS = data["PS"]
            stateNumberOfFeature = data["stateNumberOfFeature"]
            G = nx.MultiDiGraph()
            for i in range(np.prod(stateNumberOfFeature)):
                G.add_node(i)
            for i in range(len(state)):
                for j in range(len(state)):
                    if resultMatrix[i, j] != '' and i != j:
                        G.add_edge(state[i], state[j], label=resultMatrix[i, j])
            pos = nx.kamada_kawai_layout(G, scale=2)
            positions[k].append(deepcopy(pos))
            Gs[k].append(deepcopy(G))
    for i, dataType in enumerate(dataTypes):
        path = "../../MyData/transition/transitionMatrix/" + dataType + ".pkl"
        datas = pd.read_pickle(path)
        for j, data in enumerate(datas):
            print(i, j, "=" * 100)
            ax = plt.subplot(3, 3, int(i * 3) + j + 1)
            state = data["state"]
            resultMatrix = data["resultMatrix"]
            featureVector = data["featureVector"]
            PS = data["PS"]
            stateNumberOfFeature = data["stateNumberOfFeature"]
            if i == 1:
                PlotGraph2D_(positions[i + 1][j], Gs[i][j], featureVector, ax=ax)
            else:
                PlotGraph2D_(positions[i][j], Gs[i][j], featureVector, ax=ax)
    path = "../../result/transition/transition.pdf"
    plt.savefig(path)
    plt.show()


# Load d3blocks


#

class CustomD3Blocks(D3Blocks):

    def __init__(self, chart='Chord', frame=True):
        super().__init__(chart, frame)

    def add_legend(self, df):
        unique_labels = df['label'].unique()
        unique_colors = df['color'].unique()
        legend_data = list(zip(unique_labels, unique_colors))

        # 插入D3.js代码创建图例
        # 这部分代码需要根据D3Blocks的具体实现来调整
        for label, color in legend_data:
            self.chart.append('text').attr('fill', color).text(label)


def PLOTTest():
    path = "../../MyData/transition/csvData/"
    fileNames = os.listdir(path)
    filePaths = [path + fileName for fileName in fileNames]
    for filePath in filePaths:
        savepath = "../../result/transition/" + filePath.split("/")[-1][:-4] + ".html"
        df = pd.read_csv(filePath)
        d3 = D3Blocks(chart='Chord', frame=False)

        d3.set_node_properties(df, opacity=1, color="black")
        d3.set_edge_properties(df, color=df["color"], opacity='source')
        title = filePath.split("/")[-1][:-6] + "(" + df["feature"].iloc[0] + ")"
        d3.show(filepath=savepath, title=title)


strategyColor = {
    "G": "#69A89C",
    "L": "#DD8162",
    # "ev1": "#A64E5A",
    "e": "#FFD58F",
    "A": "#4E5080",
    "E": "#4F74A2",
    "N": "#7FB069",
    "S": "#8C6B93",
}


def toCsv(Matrix, featureVector, state, PS, stateNumberOfFeature, savePath):
    number = np.prod(stateNumberOfFeature)

    source = []
    target = []
    label = []
    color = []
    value = []
    temp = {}
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            if i != j and Matrix[i][j] != '':
                for s in Matrix[i][j]:
                    sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]])
                    targetNode = ",".join([str(s) for s in list(featureVector.loc[state[j]])[:-1]]) + "(0)"
                    if (sourceNode, targetNode) in temp:
                        temp[(sourceNode, targetNode)] += 1
                        sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]]) + "(" + str(
                            temp[(sourceNode, targetNode)]) + ")"
                    else:
                        temp.update({(sourceNode, targetNode): 1})
                        sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]]) + "(1)"

                    source.append(sourceNode)
                    target.append(targetNode)
                    label.append(s)
                    color.append(strategyColor[s])
                    value.append(1)
    for i in range(Matrix.shape[0]):
        temp = list(Matrix[i, :])
        temp = temp[:i] + temp[i + 1:]
        if "".join(temp) == "":
            sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]])
            targetNode = sourceNode
            source.append(sourceNode)
            target.append(targetNode)
            label.append('e')
            color.append('white')
            value.append(1)

    edges = pd.DataFrame(
        {"source": source, "target": target, "value": value, "label": label, "color": color,
         "feature": [",".join(list(featureVector.columns)[:-1])] * len(color)})
    return edges
    pass


def toCsvMain():
    # path="D:/Application/Tencent/Chats/WeChat/WeChat Files/wxid_qrarj441u6ud22/FileStorage/File/2023-08/transitionMatrix/"
    path = "../../MyData/transition/transitionMatrix/"
    fileNames = os.listdir(path)
    filePaths = [path + fileName for fileName in fileNames]
    for i, filePath in enumerate(filePaths):
        datas = pd.read_pickle(filePath)
        edges = []
        for j, data in enumerate(datas):
            state = data["state"]
            resultMatrix = data["resultMatrix"]
            featureVector = data["featureVector"]
            PS = data["PS"]
            stateNumberOfFeature = data["stateNumberOfFeature"]

            savePath = "../../MyData/transition/csvData/" + filePath.split("/")[-1][:-4] + "-" + str(j) + ".csv"
            edge = toCsv(resultMatrix, featureVector, state, PS, stateNumberOfFeature, savePath)
            edges.append(edge)
        savePath = "../../MyData/transition/csvData/" + filePath.split("/")[-1][:-4] + ".csv"
        edges = pd.concat(edges)
        edges.to_csv(savePath, encoding="UTF-8", index=False)


# def chordDiagram():
#     path = "../../MyData/transition/csvData/"
#     fileNames = os.listdir(path)
#     filePaths = [path + fileName for fileName in fileNames]
#     for filePath in filePaths:
#         savepath = "../../result/transition/" + filePath.split("/")[-1][:-4] + ".pdf"
#         df = pd.read_csv(filePath)
#         nodes = list(set(list(df["source"]) + list(df["target"])))
#         nodes.sort()
#         temp = [(i, len(nodes[i])) for i in range(len(nodes))]
#         temp = sorted(temp, key=lambda x: x[1])
#         temp = [t[0] for t in temp]
#         nodes = [nodes[t] for t in temp]
#
#         link_cmap = [(df["source"].iloc[i], df["target"].iloc[i], df["color"].iloc[i]) for i in range(len(df))]
#         matrix = Matrix.parse_fromto_table(df)
#         # 根据数量确定长度
#         degree1 = np.ceil(2 * len(df) / 110 * 99)
#         degree2 = degree1 + 1
#         # degree1 = 90
#         # degree2 = 100
#
#         plt.figure(figsize=(10, 10), dpi=300)
#         ax = plt.subplot(111, projection='polar')
#         circos = Circos.initialize_from_matrix(
#             matrix,
#             space=180 / len(df),
#             r_lim=(degree1, degree2),
#             cmap=dict({name: "white" for name in matrix.all_names}),
#             link_cmap=link_cmap,
#             label_kws=dict(size=8, r=degree2 + 5, orientation="vertical"),
#             link_kws=dict(direction=1, lw=0.05, alpha=0.6),
#             order=nodes
#         )
#
#         legends = list(set([(df["color"].iloc[i], df["label"].iloc[i]) for i in range(len(df))]))
#         legend_elements = [Patch(facecolor=le[0], edgecolor=le[0], label=le[1]) for le in legends]
#
#         # Place the legend on the plot
#         plt.legend(handles=legend_elements, loc='upper right')
#         fig = circos.plotfig(dpi=300, ax=ax)
#         plt.savefig(savepath)
#         plt.show()


def chordDiagram():
    path = "../../MyData/transition/csvData/"
    fileNames = os.listdir(path)
    filePaths = [path + fileName for fileName in fileNames]
    for j, filePath in enumerate(filePaths):
        # if j < 2:
        #     continue
        savepath = "../../result/transition/" + filePath.split("/")[-1][:-4] + ".pdf"
        df = pd.read_csv(filePath)
        nodes = list(set(list(df["source"]) + list(df["target"])))
        nodes.sort()
        temp = [(i, len(nodes[i])) for i in range(len(nodes))]
        temp = sorted(temp, key=lambda x: x[1])
        temp = [t[0] for t in temp]
        nodes = [nodes[t] for t in temp]

        sapce = [180 / len(df) if i == len(nodes) - 1 or nodes[i][:-5] != nodes[i + 1][:-5] else 0 for i in
                 range(len(nodes))]

        sapce = [180 / len(df) if i == len(nodes) - 1 or nodes[i][:-3] != nodes[i + 1][:-3] else 0 for i in
                 range(len(nodes))]

        link_cmap = [(df["source"].iloc[i], df["target"].iloc[i], df["color"].iloc[i]) for i in range(len(df))]
        matrix = Matrix.parse_fromto_table(df)

        degree1 = 98
        degree2 = 100

        plt.figure(figsize=(10, 10), dpi=300)
        ax = plt.subplot(111, projection='polar')
        circos = Circos.initialize_from_matrix(
            matrix,
            space=sapce,
            r_lim=(degree1, degree2),
            cmap=dict({name: "black" for name in matrix.all_names}),
            link_cmap=link_cmap,
            label_kws=dict(size=8, r=degree2 + 5, orientation="vertical"),
            link_kws=dict(direction=1, lw=0.05, alpha=0.6),
            order=nodes
        )

        for sector in circos.sectors:
            for tractor in sector.tracks:
                tractor.axis(fc="black", ec="black")

        legends = list(set([(df["color"].iloc[i], df["label"].iloc[i]) for i in range(len(df))]))
        legends = [('#69A89C', 'gl'), ('#DD8162', 'lo'), ('#FFD58F', 'ev'), ('#4E5080', 'ap'), ('#4F74A2', 'en'),
                   ('#7FB069', 'sv'), ('#8C6B93', 'st')]
        legend_elements = [Patch(facecolor=le[0], edgecolor=le[0], label=le[1]) for le in legends]

        # Place the legend on the plot
        font_properties = {'family': 'CMU Serif', 'size': 20}
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.9, 1.1), prop=font_properties)
        fig = circos.plotfig(dpi=300, ax=ax)
        # plt.savefig(savepath)
        plt.show()


def heatMap():
    pass


if __name__ == '__main__':
    static()
    toCsvMain()
    chordDiagram()
    # PLOTTest()

    # PlotMain()
