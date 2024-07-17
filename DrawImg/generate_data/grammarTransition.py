# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from pycirclize.parser import Matrix
import matplotlib.pyplot as plt
from multiprocessingExecutor import ParallelExecutor
from pycirclize import Circos
from matplotlib.patches import FancyArrowPatch
import random
import colorsys
from itertools import accumulate
import seaborn as sns
from copy import deepcopy

plt.rcParams['axes.unicode_minus'] = False

novice_index = [0, 9, 14, 16, 24, 26, 32]
expert_index = list(set(list(range(34))) - set(novice_index))
expert_index.sort()
parallelExecutor = ParallelExecutor()

grammars_labels = ["$p_u$", "$p_b$", "$g_u$", "$g_b$", "$h_p$", "$h_g$", "$sv$", "$e$"]
grammar_colors = ["#C0FFC0", "#0edb0e", "#ADD8E6", "#0000CD", "#D8BFD8", "#9400D3", "#7FB069", "#FFD58F"]


def one_hot_to_number(state):
    state = np.array(state)
    factor = np.array([72, 36, 12, 4, 2, 1])
    return np.sum(state * factor)


def gram_type(gram, grammarLen):
    grammar_types = {
        "L": [0], "G": [0],
        "1": [7], "2": [7],
        "A": [2], "E": [2], "S": [2],
        "GL": [1], "LG": [1], "LE": [1],
        "AL": [1], "EA": [3],
        "LEA": [4], "EAL": [4], "EAG": [4], "EAGL": [4],
        "SEA": [4],
    }
    if gram in grammar_types.keys():
        return grammar_types[gram]
    elif "N" in gram and grammarLen > 1:
        return [6]
    else:
        return [8]

    # grammar_types = {
    #     "L": [0], "G": [0],
    #     "1": [6], "2": [6],
    #     "A": [3], "E": [3], "S": [3],
    #     "GL": [1], "LG": [1], "LE": [1],
    #     "AL": [1, 4], "EA": [4],
    #     "LEA": [2, 5], "EAL": [2, 5], "EAG": [2, 5], "EAGL": [2, 5],
    #     "SEA": [5],
    # }
    # if gram in grammar_types.keys():
    #     return grammar_types[gram]
    # elif "N" in gram and grammarLen > 1:
    #     return [5]
    # else:
    #     return [7]


def generate_transition_data(paths):
    grammar_path = paths[0]
    print(grammar_path)
    state_path = paths[1]
    grammar_data = pd.read_pickle(grammar_path)[["Unnamed: 0", "gram", "gramLen"]]
    state_data = pd.read_pickle(state_path)[["Unnamed: 0", "PG1", "PG2", "IS1", "IS2", "PE", "BN5"]]

    # state_data = state_data.apply(lambda x: one_hot_to_number(x), axis=1)

    data = pd.merge(grammar_data, state_data, how="left", on="Unnamed: 0")

    data = data[data["gram"] != data["gram"].shift(1)]
    grammar = data[["gram", "gramLen"]].apply(lambda x: gram_type(x.gram, x.gramLen), axis=1)
    state = data[["PG1", "PG2", "IS1", "IS2", "PE", "BN5"]].apply(lambda x: one_hot_to_number(x), axis=1)

    feature_vector = data[["PG1", "PG2", "IS1", "IS2", "PE", "BN5"]]
    feature_vector["state"] = state
    feature_vector = feature_vector.drop_duplicates()

    transition = pd.DataFrame(
        {"s": state[:-1], "g": grammar[:-1], "s'": state.shift(-1)[:-1], }
    )
    # transition = transition[transition["g"].apply(lambda gs: not all(x == 7 for x in gs))]
    return transition, feature_vector


def generate_transition_data_human(Type):
    if Type == "expert" or Type == "novice":
        grammar_path = "../../../Monkey_Analysis/fmri_data_process/fmriGramDataGhost2/"
        state_path = "../../../Monkey_Analysis/fmri_data_process/fmriDiscreteFeatureData/"
        grammar_filenames = os.listdir(grammar_path)
        file_paths = [grammar_path + filename for filename in grammar_filenames]
        state_filenames = os.listdir(state_path)
        state_file_paths = [state_path + filename for filename in state_filenames]
        paths = [(file_paths[i], state_file_paths[i]) for i in range(len(file_paths))]
        if Type == "expert":
            paths = [paths[i] for i in range(len(paths)) if i in expert_index]
        else:
            paths = [paths[i] for i in range(len(paths)) if i in novice_index]
    elif Type == "monkey":
        grammar_path = "../../../Monkey_Analysis/MonkeyData/MonkeyGram/data20-21_every_monkey/"
        state_path = "../../../Monkey_Analysis/MonkeyData/MonkeyDiscreteFeature/data20-21/"

        state_filenames = os.listdir(state_path)
        state_file_paths = [state_path + filename for filename in state_filenames]
        state_file_key = ["-".join(filename.split("-")[:5]) for filename in state_filenames]

        grammar_filenames = os.listdir(grammar_path)
        file_paths = [grammar_path + filename for filename in grammar_filenames if
                      filename.split("/")[-1][:-4] in state_file_key]

        state_file_paths.sort()
        file_paths.sort()
        paths = [(file_paths[i], state_file_paths[i]) for i in range(len(file_paths))]
    print(len(paths))
    # result = []
    # for i in range(2):
    #     result.append(generate_transition_data(paths[i]))
    result = parallelExecutor.ParallelExecute(generate_transition_data, paths)

    transition = [r[0] for r in result]
    feature_vector = [r[1] for r in result]
    transition = pd.concat(transition)
    feature_vector = pd.concat(feature_vector)
    feature_vector = feature_vector.drop_duplicates()

    result = {
        "transition": transition,
        "feature_vector": feature_vector
    }
    save_path = "../../MyData/grammarTransition/transition_" + Type + ".pkl"

    pd.to_pickle(result, save_path)


def flatten_transition(t):
    grammar = t["g"]
    s = t["s"]
    s_ = t["s'"]
    transition = []
    for g in grammar:
        transition.append((s, g, s_))
    return transition


def generate_color(num):

    random.seed(num)


    h = random.random()
    s = 0.5 + random.random() * 0.5
    v = 0.5 + random.random() * 0.5


    r, g, b = [int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v)]


    return f'#{r:02X}{g:02X}{b:02X}'


def PlotPS(probabilities, threshold=0.7, show=False):

    index = np.argsort(probabilities)[::-1]
    probabilities = probabilities[index]
    options = index
    colors = [generate_color(i) for i in index]

    if show == True:

        fig, ax = plt.subplots()

        bottom = 0

        for i in range(len(probabilities)):
            ax.bar(1, probabilities[i], bottom=bottom, color=colors[i], edgecolor='white', label=options[i])
            bottom += probabilities[i]

        ax.set_ylabel('Probability')
        ax.set_title('Probabilities of state')
        ax.set_xticks([1])
        ax.set_xticklabels([''])
        ax.axhline(y=threshold, color='grey', linestyle='--', linewidth=1)

        plt.show()

    prefix_sum = list(accumulate(probabilities))
    end = 0
    for i in range(len(prefix_sum)):
        if prefix_sum[i] > threshold:
            end = i - 1
            break
        print(i, prefix_sum[i])

    return np.sort(index[:end])


def find_outliers(arr, num_std_devs=3):

    mean = np.mean(arr)
    std_dev = np.std(arr)

    outliers = []
    for x in arr:
        if x > mean + num_std_devs * std_dev:
            outliers.append(1)
        else:
            outliers.append(0)

    return outliers


def plot_state_transition(data, state, show=False):
    data = pd.DataFrame(data, columns=state, index=state)
    if show == True:
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data, cmap='PuBuGn',
                    linewidths=.1,
                    fmt='.2f',
                    annot=True, )
        plt.yticks(rotation=0)
        plt.show()
    result = []
    for i in range(len(data)):
        d = data.iloc[i]
        d = find_outliers(d, num_std_devs=1.2)
        result.append(d)
    result = np.array(result)
    data = pd.DataFrame(result, columns=state, index=state)
    if show == True:
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(data, cmap='PuBuGn',
                    linewidths=.1,
                    annot=True, )
        plt.yticks(rotation=0)
        plt.show()
    return data.values


def save_all_state(threshold, show):
    def get_key(x):
        key = np.array(x)[:-1]
        change_type = {0: 0, 1: 4, 2: 1}
        key[2] = change_type[key[2]]
        key[3] = change_type[key[3]]
        key = [key[2] + key[3]] + list(key[[0, 1, 4, 5]])
        key = "".join([str(k) for k in key])
        return key

    subj_types = ["expert", "novice", "monkey"]
    save_states = []
    feature_vectors = []
    probabilities = []

    for subj in subj_types:
        print(subj)
        transition_result = pd.read_pickle("../../MyData/grammarTransition/transition_" + subj + ".pkl")
        transition = transition_result["transition"]
        feature_vector = transition_result["feature_vector"]
        feature_vector = feature_vector.sort_values("state")

        # select state
        state = transition["s"]
        probability_of_s = np.bincount(transition["s"]) / len(state)
        probabilities.append(probability_of_s)
        save_state = PlotPS(probability_of_s, threshold, show)

        print(len(save_state))
        feature_vector = feature_vector[feature_vector["state"].isin(save_state)]
        feature_vector.index = feature_vector.state
        save_states.append(save_state)
        feature_vectors.append(feature_vector)
    feature_vector = pd.concat(feature_vectors).drop_duplicates()
    feature_vector["key"] = feature_vector.apply(lambda x: get_key(x), axis=1)
    feature_vector = feature_vector.sort_values("key")
    
    row = feature_vector.loc[71]
    feature_vector = feature_vector.drop(71)
    feature_vector = pd.concat([feature_vector, pd.DataFrame([row])])
    # feature_vector = feature_vector.append(row)
    num = list(range(1, len(feature_vector) + 1))
    num = ["0" * (2 - len(str(n))) + str(n) for n in num]
    feature_vector["num"] = num

    #
    font_properties = {'family': 'CMU Serif', 'size': 20}
    fig = plt.figure(figsize=(10, 8), dpi=300)
    colors = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]
    for i in range(len(probabilities)):
        probabilities[i] = np.sort(probabilities[i])[::-1][:20]
        probabilities[i] = list(accumulate(probabilities[i]))
        probabilities[i] = np.log(probabilities[i])
        plt.plot(probabilities[i], color=colors[i], linewidth=2)
    plt.axhline(y=np.log(0.7), color='grey', linestyle='--', linewidth=2)
    plt.legend(["Expert", "Novice", "Monkey"], prop=font_properties, bbox_to_anchor=(0.7, 0.5))
    plt.xlabel("number of states", fontdict=font_properties)
    plt.ylabel("log cummutive probability of state", fontdict=font_properties)
    plt.xticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.show()
    return feature_vector


def feature_vector_name(state):
    state = list(state)
    change_name = [
        {0: "N", 1: "D", 2: "S"}, {0: "N", 1: "D", 2: "S"}, {0: "C", 1: "F"}, {0: "C", 1: "F"}, {0: "C", 1: "F"},
        {0: "N", 1: "A"},
    ]
    state_value_name = [change_name[i][state[i]] for i in range(len(state))]
    return state_value_name


def transition_filter(threshold=0.7, show=False):
    state_vector = save_all_state(threshold, False)
    temp = deepcopy(state_vector[["IS1", "IS2", "PG1", "PG2", "PE", "BN5", "num"]])
    temp["state"] = temp[["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]].apply(lambda x: feature_vector_name(x), axis=1)
    temp["state"] = temp["state"].apply(lambda x: "-".join(list(x)))
    temp.rename(columns={
        "IS1": "m1", "IS2": "m2", "PG1": "g1", "PG2": "g2", "PE": "e", "BN5": "b"
    }, inplace=True)

    subj_types = ["expert", "novice", "monkey"]
    for subj in subj_types[:]:
        print(subj)
        transition_result = pd.read_pickle("../../MyData/grammarTransition/transition_" + subj + ".pkl")
        transition = transition_result["transition"]

        state = transition["s"]
        probability_of_s = np.bincount(transition["s"]) / len(state)
        save_state = PlotPS(probability_of_s, threshold, show)
        save_state = [s for s in state_vector["state"] if s in save_state]
        feature_vector = deepcopy(state_vector[state_vector["state"].isin(save_state)])

        state_transition_matrix = np.zeros((len(probability_of_s), len(probability_of_s)))
        for idx, grp in transition.groupby(["s", "s'"]):
            s = int(idx[0])
            s_ = int(idx[1])
            if s != s_:
                state_transition_matrix[s, s_] = len(grp)
        state_transition_matrix = state_transition_matrix[save_state, :][:, save_state]
        for i in range(len(state_transition_matrix)):
            state_transition_matrix[i] = state_transition_matrix[i] / np.sum(state_transition_matrix[i])
        state_name = ["".join([str(ss) for ss in list(
            feature_vector.loc[s][["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]])]) + "(" + str(
            feature_vector.loc[s]["num"]) + ")" for s in
                      save_state]
        mask = plot_state_transition(state_transition_matrix, state_name, show)

        delete_num = 8
        grammar = np.array(sum([list(g) for g in list(transition["g"])], []))
        grammar = [g for g in grammar if g != delete_num]
        probability_of_g = np.bincount(grammar) / len(grammar)

        new_transition = transition.apply(lambda x: flatten_transition(x), axis=1)
        new_transition = pd.DataFrame(sum(new_transition, []), columns=["s", "g", "s'"])
        new_transition = new_transition[new_transition["g"] != delete_num]
        new_transition = new_transition[new_transition["s"].isin(save_state)]
        new_transition = new_transition[new_transition["s'"].isin(save_state)]

        transitionMatrix = np.zeros((len(probability_of_g), len(probability_of_s), len(probability_of_s)))
        for idx, grp in new_transition.groupby(["s", "g", "s'"]):
            s = int(idx[0])
            s_ = int(idx[2])
            g = int(idx[1])
            if s == 37 and s_ == 109:
                x = 0
            transitionMatrix[g, s, s_] = len(grp)
        transitionMatrix = transitionMatrix[:, save_state, :][:, :, save_state]
        transitionMatrix[:, mask == 0] = 0

        # P(s'|s,a)
        CPS_ = np.zeros(transitionMatrix.shape)
        for k in range(transitionMatrix.shape[0]):
            for l in range(transitionMatrix.shape[1]):
                temp = np.sum(transitionMatrix[k, l, :])
                for m in range(transitionMatrix.shape[2]):
                    if temp != 0:
                        CPS_[k, l, m] = transitionMatrix[k, l, m] / temp
        # P(a|s,s')
        CPA = np.zeros(transitionMatrix.shape)
        for l in range(transitionMatrix.shape[1]):
            for m in range(transitionMatrix.shape[2]):
                temp = np.sum(transitionMatrix[:, l, m])
                for k in range(transitionMatrix.shape[0]):
                    if temp != 0:
                        CPA[k, l, m] = transitionMatrix[k, l, m] / temp

        resultMatrix = [["" for j in range(len(save_state))] for i in range(len(save_state))]
        results = []
        for i in range(transitionMatrix.shape[1]):
            for j in range(transitionMatrix.shape[2]):
                temp = transitionMatrix[:, i, j]
                if np.sum(temp) != 0:
                    max_value = np.max(temp)
                    indexs = np.where(temp == max_value)[0]
                    # indexs = np.array(range(len(temp)))
                    indexs = np.argsort(temp)[::-1][:1]
                    # print(temp[indexs] / len(new_transition))
                    # if temp[indexs[1]] < 40:
                    #     indexs = indexs[:1]
                    # if temp[indexs[1]] / len(new_transition) < 0.0025 or temp[indexs[1]] < 10:
                    #     indexs = indexs[:1]
                    print(temp[indexs])
                    indexs = [str(int(idx)) for idx in indexs]

                    s = save_state[i]
                    s_ = save_state[j]
                    for g in indexs:
                        g = int(g)
                        if transitionMatrix[g, i, j] < 5:
                            continue
                        resultMatrix[i][j] += str(g)

                        source_num = feature_vector.loc[s]["num"]
                        target_num = feature_vector.loc[s_]["num"]
                        source_vectir = feature_vector_name(
                            list(feature_vector.loc[s][["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]]))
                        target_vectir = feature_vector_name(
                            list(feature_vector.loc[s_][["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]]))
                        g_name = ["pu", "pb", "gu", "gb", "hp", "hg", "sv", " e"][g]
                        num = str(int(transitionMatrix[g, i, j]))
                        if len(num) < 2:
                            num = "0" * (3 - len(num)) + num
                        results.append(
                            [source_num, g_name, target_num, num, source_vectir, target_vectir])
        results = sorted(results, key=lambda x: x[3])[::-1]
        for r in results:
            print(r)

        resultMatrix = np.array(resultMatrix)

        result = {
            "stateNumberOfFeature": [2, 2, 3, 3, 2, 2],
            "resultMatrix": resultMatrix,
            "featureVector": feature_vector,
            "state": save_state,
        }
        save_path = "../../MyData/grammarTransition/result_matrix_" + subj + ".pkl"
        pd.to_pickle(result, save_path)


def toCsv(Matrix, featureVector, state, stateNumberOfFeature):
    labels = ["$p_u$", "$p_b$", "$p_t$", "$g_u$", "$g_b$", "$g_t$", "$e$"]
    strategyColor = ["#90EE90", "#00FF00", "#006400", "#ADD8E6", "#0000FF", "#00008B", "#FFD58F"]

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
                    # sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]])
                    # targetNode = ",".join([str(s) for s in list(featureVector.loc[state[j]])[:-1]]) + "(0)"

                    sourceNode = featureVector.loc[state[i]]["num"]
                    targetNode = featureVector.loc[state[j]]["num"] + "(0)"
                    if (sourceNode, targetNode) in temp:
                        temp[(sourceNode, targetNode)] += 1
                        sourceNode = featureVector.loc[state[i]]["num"] + "(" + str(
                            temp[(sourceNode, targetNode)]) + ")"
                    else:
                        temp.update({(sourceNode, targetNode): 1})
                        sourceNode = featureVector.loc[state[i]]["num"] + "(1)"

                    source.append(sourceNode)
                    target.append(targetNode)
                    label.append(grammars_labels[int(s)])
                    color.append(grammar_colors[int(s)])
                    value.append(1)
    # for i in range(Matrix.shape[0]):
    #     temp = list(Matrix[i, :])
    #     temp = temp[:i] + temp[i + 1:]
    #     if "".join(temp) == "":
    #         sourceNode = ",".join([str(s) for s in list(featureVector.loc[state[i]])[:-1]])
    #         targetNode = sourceNode
    #         source.append(sourceNode)
    #         target.append(targetNode)
    #         label.append('e')
    #         color.append('white')
    #         value.append(1)

    edges = pd.DataFrame(
        {"source": source, "target": target, "value": value, "label": label, "color": color,
         "feature": [",".join(list(featureVector.columns)[:-1])] * len(color)})
    return edges


def toCsvMain():
    path = "../../MyData/grammarTransition/"
    subj_types = ["expert", "novice", "monkey"]
    for i, subj in enumerate(subj_types):
        result = pd.read_pickle(path + "result_matrix_" + subj + ".pkl")
        state = result["state"]
        resultMatrix = result["resultMatrix"]
        # temp=pd.DataFrame(resultMatrix)
        featureVector = result["featureVector"][["IS1", "IS2", "PG1", "PG2", "PE", "BN5", "state", "num"]]
        stateNumberOfFeature = result["stateNumberOfFeature"]
        edge = toCsv(resultMatrix, featureVector, state, stateNumberOfFeature)
        save_path = "../../MyData/grammarTransition/transition_csv_" + subj + ".csv"
        edge.to_csv(save_path, encoding="UTF-8", index=False)


def chordDiagram():
    path = "../../MyData/grammarTransition/"
    subj_types = ["expert", "novice", "monkey"]

    for j, subj in enumerate(subj_types):
        data_path = path + "transition_csv_" + subj + ".csv"
        df = pd.read_csv(data_path)
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
            label_kws=dict(size=14, r=degree2 + 5, orientation="vertical"),
            link_kws=dict(direction=1, lw=0.05, alpha=0.6),
            order=nodes
        )

        for sector in circos.sectors:
            for tractor in sector.tracks:
                tractor.axis(fc="black", ec="black")
        # "#008000",


        legend_elements = [
            FancyArrowPatch((0, 0), (1, 1), color=color, arrowstyle='->', mutation_scale=15, label=label)
            for color, label in zip(grammar_colors, grammars_labels)
        ]

        # Place the legend on the plot
        font_properties = {'family': 'CMU Serif', 'size': 15}
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1.17), prop=font_properties)
        fig = circos.plotfig(dpi=300, ax=ax)
        # plt.savefig(savepath)
        plt.show()


if __name__ == '__main__':
    pass
    # subj_types = ["expert", "novice", "monkey"]
    # for subj in subj_types:
    #     generate_transition_data_human(subj)
    # transition_filter(0.7, False)
    # toCsvMain()
    # chordDiagram()
