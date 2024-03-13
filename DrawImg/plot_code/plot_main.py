import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import plotly.tools as tls
import plotly.offline
import plotly.graph_objects as go
import plotly
import plotly.io as pio
from matplotlib.patches import Patch
import scipy.stats as stats
from utility import significant, violin, violin_monkey, draw_state
from sklearn.linear_model import LinearRegression
import pingouin as pg
import pickle
from pycirclize.parser import Matrix
from pycirclize import Circos
import seaborn as sns


def plot_2b():
    data = pd.read_csv("../plot_data/Fig2b.csv")
    x = data[['ratio of uni-gram', 'ratio of bi-gram', 'ratio of tri-gram and skip-gram']].values * 100
    y = np.array(data["labels"])
    colors = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]
    legends = ["Expert", "Amateur", "Monkey O", "Monkey P"]
    fig = plt.figure(figsize=(17, 17), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    for label, color in enumerate(colors):
        indices = np.where(y == label)
        print(np.mean(x[indices[0], :], axis=0))
        # 3D Scatter
        ax.scatter(x[indices, 0], x[indices, 1], x[indices, 2], color=color, s=200, label=legends[label])
        # Calculate and draw the centroid projection line
        centroid = x[indices].mean(axis=0)
        ax.plot([centroid[0], centroid[0]], [centroid[1], 0], [centroid[2], centroid[2]], linestyle='--', color=color)
        # Draw a transparent sphere and filter the parts where the z-axis is less than 0
        radius = np.max(np.linalg.norm(x[indices] - centroid, axis=1))
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        X = centroid[0] + radius * np.outer(np.cos(u), np.sin(v))
        Y = centroid[1] + radius * np.outer(np.sin(u), np.sin(v))
        Z = centroid[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        index = np.where(Z >= 0)
        newX = []
        newY = []
        newZ = []
        for idx in range(len(index[0])):
            x1 = index[0][idx]
            y1 = index[1][idx]
            if len(newX) <= x1:
                newX.append([])
                newY.append([])
                newZ.append([])
            newX[x1].append(X[x1][y1])
            newY[x1].append(Y[x1][y1])
            newZ[x1].append(Z[x1][y1])
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.2, shade=False)

    # Set the background grid line color to white
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True

    axisLabel = ['ratio of uni-gram', 'ratio of bi-gram', 'ratio of n-gram(n>2)']
    font_properties = {'family': 'CMU Serif', 'size': 40}
    ax.view_init(elev=20, azim=30)
    ax.set_xlabel(axisLabel[0], fontproperties=font_properties, labelpad=30)
    ax.set_ylabel(axisLabel[1], fontproperties=font_properties, labelpad=30)
    ax.set_zlabel(axisLabel[2], fontproperties=font_properties, labelpad=20)

    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_zticks([0, 10, 20, 30])
    ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70], fontdict=font_properties)
    ax.set_yticklabels([0, 10, 20, 30, 40, 50], fontdict=font_properties)
    ax.set_zticklabels([0, 10, 20, 30], fontdict=font_properties)
    font_properties["size"] = 30
    ax.legend(bbox_to_anchor=(0.95, 0.8), prop=font_properties)
    plt.tight_layout()
    plt.subplots_adjust(left=0.012, right=1, bottom=0, top=1)
    plt.savefig(".../plot_img/Fig2b.pdf")
    plt.show()

def plot_2c():
    y_data = pd.read_pickle("../plot_data/Fig2c.pkl")
    y_data = [y_data["Expert"], y_data["Novice"], y_data["Monkey O"], y_data["Monkey P"]]
    POSITIONS = [i * 0.8 for i in range(len(y_data))]
    x_data = [np.array([POSITIONS[i]] * len(m)) for i, m in enumerate(y_data)]
    jitter = 0.04
    x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    # Colors
    BLACK = "#282724"
    # Colors taken from Dark2 palette in RColorBrewer R library
    COLOR_SCALE = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]
    font_properties = {'family': 'CMU Serif', 'size': 40}

    fig, ax = plt.subplots(figsize=(14, 17), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Add jittered dots ----------------------------------------------
    for x, y, color in zip(x_jittered, y_data, COLOR_SCALE):
        ax.scatter(x, y, s=100, color=color, alpha=0.4)
    # Add violins
    violins = ax.violinplot(
        y_data,
        positions=POSITIONS,
        widths=0.45,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    # Customize violins (remove fill, customize line, etc.)
    for pc in violins["bodies"]:
        pc.set_facecolor("none")
        pc.set_edgecolor(BLACK)
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    # Add boxplots
    medianprops = dict(
        linewidth=4,
        color=BLACK,
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=2,
        color=BLACK
    )
    ax.boxplot(
        y_data,
        positions=POSITIONS,
        showfliers=False,  # Do not show the outliers beyond the caps.
        showcaps=False,  # Do not show the caps
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )
    # Add mean value labels ------------------------------------------
    means = [y.mean() for y in y_data]
    for i, mean in enumerate(means):
        # Add dot representing the mean
        ax.scatter(POSITIONS[i], mean, s=250, color="black", zorder=3)

    tick_len = 0.01
    p1 = 3.3
    p2 = 3.2
    p3 = 3.1
    p4 = 3.5
    p5 = 3.4
    p6 = 3.6

    ax.plot([POSITIONS[0], POSITIONS[0], POSITIONS[1], POSITIONS[1]], [p1 - tick_len, p1, p1, p1 - tick_len], c="black")
    ax.plot([POSITIONS[1], POSITIONS[1], POSITIONS[2], POSITIONS[2]], [p2 - tick_len, p2, p2, p2 - tick_len], c="black")
    ax.plot([POSITIONS[2], POSITIONS[2], POSITIONS[3], POSITIONS[3]], [p3 - tick_len, p3, p3, p3 - tick_len], c="black")
    ax.plot([POSITIONS[0], POSITIONS[0], POSITIONS[2], POSITIONS[2]], [p4 - tick_len, p4, p4, p4 - tick_len], c="black")
    ax.plot([POSITIONS[1], POSITIONS[1], POSITIONS[3], POSITIONS[3]], [p5 - tick_len, p5, p5, p5 - tick_len], c="black")
    ax.plot([POSITIONS[0], POSITIONS[0], POSITIONS[3], POSITIONS[3]], [p6 - tick_len, p6, p6, p6 - tick_len], c="black")

    # Add labels for the p-values
    significant_combinations = significant(y_data)
    ps = []
    for sig in significant_combinations:
        if sig[1] < 0.001:
            sig_symbol = '***'
        elif sig[1] < 0.01:
            sig_symbol = '**'
        else:
            sig_symbol = "N"
        ps.append(sig_symbol)
    label1 = ps[3]
    label2 = ps[4]
    label3 = ps[5]
    label4 = ps[1]
    label5 = ps[2]
    label6 = ps[0]

    pad = 0.015
    ax.text(0.5 * (POSITIONS[1] - POSITIONS[0]), p1 - pad, label1, va="bottom", ha="center", fontdict=font_properties)
    ax.text(0.75 * (POSITIONS[2] - POSITIONS[0]), p2 - pad, label2, va="bottom", ha="center", fontdict=font_properties)
    ax.text(5 / 6 * (POSITIONS[3] - POSITIONS[0]), p3 - pad, label3, va="bottom", ha="center", fontdict=font_properties)
    ax.text(0.5 * (POSITIONS[2] - POSITIONS[0]), p4 - pad, label4, va="bottom", ha="center", fontdict=font_properties)
    ax.text(4 / 6 * (POSITIONS[3] - POSITIONS[0]), p5 - pad, label5, va="bottom", ha="center", fontdict=font_properties)
    ax.text(0.5 * (POSITIONS[3] - POSITIONS[0]), p6 - pad, label6, va="bottom", ha="center", fontdict=font_properties)

    plt.xticks([0, 1, 2, 3],
               ["Expert\n" + "(n=" + str(len(y_data[0])) + ")", "Amateur\n" + "(n=" + str(len(y_data[1])) + ")",
                "Monkey O\n" + "(n=" + str(len(y_data[2])) + ")", "Monkey P\n" + "(n=" + str(len(y_data[3])) + ")"])

    plt.xticks(POSITIONS, ["Expert", "Novice", "Monkey O", "Monkey P"], fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.ylabel("LoPS complexity", fontdict=font_properties)
    plt.tight_layout()
    # path = "../../result/Fig-gram-depth.pdf"
    plt.savefig("../plot_img/Fig2c.pdf")
    plt.show()

def plot_3a():
    data = pd.read_pickle("../plot_data/Fig3a.pkl")
    ExpertGramDepth = data["ExpertGramDepth"]
    ExpertReactionTime = data["ExpertReactionTime"]
    AmateurGramDepth = data["AmateurGramDepth"]
    AmateurReactionTime = data["AmateurReactionTime"]
    MonkeyGramDepthO = data["MonkeyGramDepth O"]
    MonkeyReactionTimeO = data["MonkeyReactionTime O"]
    MonkeyGramDepthP = data["MonkeyGramDepth P"]
    MonkeyReactionTimeP = data["MonkeyReactionTime P"]

    colors = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]

    font_properties = {'family': 'CMU Serif', 'size': 30}
    fig = plt.figure(figsize=(18, 12), dpi=300)
    ax = plt.subplot(1, 1, 1)

    plt.scatter(ExpertGramDepth, ExpertReactionTime, color=colors[0], label="Expert", s=250)
    plt.scatter(AmateurGramDepth, AmateurReactionTime, color=colors[1], label="Novice", s=250)
    plt.scatter(MonkeyGramDepthO, MonkeyReactionTimeO, color=colors[2], label="Monkey O", s=250)
    plt.scatter(MonkeyGramDepthP, MonkeyReactionTimeP, color=colors[3], label="Monkey P", s=250)
    plt.legend(prop=font_properties)

    gramdepth = list(ExpertGramDepth) + list(AmateurGramDepth) + list(MonkeyGramDepthO) + list(MonkeyGramDepthP)
    meanReactionTime = list(ExpertReactionTime) + list(AmateurReactionTime) + list(MonkeyReactionTimeO) + list(
        MonkeyReactionTimeP)

    gramdepth = np.array(gramdepth).reshape(-1, 1)
    meanReactionTime = np.array(meanReactionTime).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(gramdepth, meanReactionTime)
    x = np.linspace(np.min(gramdepth), np.max(gramdepth), 1000).reshape(-1, 1)
    y = lr.predict(x)

    plt.plot(x[:, 0], y[:, 0], color="black")
    plt.xlabel("LoPS complexity", fontdict=font_properties)
    plt.ylabel("RT (ms)", fontdict=font_properties)
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    x = pg.corr(gramdepth[:, 0], meanReactionTime[:, 0] / 120)
    plt.text(1.9, 600, "r={:.2f}\nP={:.3f}".format(x.r[0], x["p-val"][0]), fontdict=font_properties)
    plt.tight_layout()
    # plt.savefig("../plot_img/Fig3a.pdf")

    plt.show()

def plot_3b():
    data = pd.read_pickle("../plot_data/Fig3b-.pkl")
    ExpertGramDepth = data["ExpertGramDepth"]
    ExpertReward = data["ExpertReward"]
    AmateurGramDepth = data["AmateurGramDepth"]
    AmateurReward = data["AmateurReward"]
    MonkeyGramDepthOmega = data["MonkeyGramDepth O"]
    MonkeyRewardOmega = data["MonkeyReward O"]
    MonkeyGramDepthPatamon = data["MonkeyGramDepth P"]
    MonkeyRewardPatamon = data["MonkeyReward P"]

    fig = plt.figure(figsize=(18, 12), dpi=300)
    ax = plt.subplot(1, 1, 1)
    font_properties = {'family': 'CMU Serif', 'size': 30}
    colors = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]
    plt.scatter(ExpertGramDepth, ExpertReward, color=colors[0], label="Expert", s=250)
    plt.scatter(AmateurGramDepth, AmateurReward, color=colors[1], label="Novice", s=250)
    plt.scatter(MonkeyGramDepthOmega, MonkeyRewardOmega, color=colors[2], label="Monkey O", s=250)
    plt.scatter(MonkeyGramDepthPatamon, MonkeyRewardPatamon, color=colors[3], label="Monkey P", s=250)
    plt.legend()

    gramdepth = list(ExpertGramDepth) + list(AmateurGramDepth) + list(MonkeyGramDepthOmega) + list(
        MonkeyGramDepthPatamon)
    reward = list(ExpertReward) + list(AmateurReward) + list(MonkeyRewardOmega) + list(
        MonkeyRewardPatamon)

    gramdepth = np.array(gramdepth).reshape(-1, 1)
    reward = np.array(reward).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(gramdepth, reward)
    x = np.linspace(np.min(gramdepth), np.max(gramdepth), 1000).reshape(-1, 1)
    y = lr.predict(x)

    plt.plot(x[:, 0], y[:, 0], color="black")

    plt.xlabel("LoPS complexity", fontdict=font_properties)
    plt.ylabel("total score (points)", fontdict=font_properties)
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    plt.legend(prop=font_properties)
    x = pg.corr(gramdepth[:, 0], reward[:, 0])
    plt.text(1.9, 30, "r={:.2f}\nP={:.3f}".format(x.r[0], x["p-val"][0]), fontdict=font_properties)

    plt.tight_layout()
    # plt.savefig("../plot_img/Fig3b.pdf")
    plt.show()

def plot_4c2():
    data = pd.read_pickle("../plot_data/Fig4c2.pkl")

    y_data = [data["session1_data"], data["session2_data"]]
    means = data["means"]
    save_path = "../plot_img/Fig_4c2.pdf"
    violin(y_data, savePath=save_path, COLOR_SCALE=["#E0E0E0", "#A0A0A0"], means=means,
           sub_color="#B4585F")

def plot_4c3():
    data = pd.read_pickle("../plot_data/Fig4c3.pkl")

    y_data = [data["session1_data"], data["session2_data"]]
    means = data["means"]
    save_path = "../plot_img/Fig_4c3.pdf"
    violin(y_data, savePath=save_path, COLOR_SCALE=["#E0E0E0", "#A0A0A0"], means=means,
           sub_color="#B4585F", layer=2)

def plot_4a2_4b2():
    data = pd.read_pickle("../plot_data/Fig4a2-4b2.pkl")
    y_data_Patamon = data["y_data_Patamon"]
    y_data_Omega = data["y_data_Omega"]

    ylabel = "LoPS Complexity"
    colors = ["#E0E0E0", "#A0A0A0", "#606060"]

    sub_color = "#8080b6"
    path = "../plot_img/Fig4b2.pdf"
    violin_monkey(y_data_Patamon, savePath=path, COLOR_SCALE=colors, sub_color=sub_color,
                  ylabel=ylabel, Type=2)

    sub_color = "#F6D087"
    path = "../plot_img/Fig4a2.pdf"
    violin_monkey(y_data_Omega, savePath=path, COLOR_SCALE=colors, sub_color=sub_color,
                  ylabel=ylabel, Type=2)

def plot_4a3_4b3():
    data = pd.read_pickle("../plot_data/Fig4a3-4b3.pkl")
    y_data_Patamon = data["y_data_Patamon"]
    y_data_Omega = data["y_data_Omega"]

    ylabel = "LoPS Complexity"
    colors = ["#E0E0E0", "#A0A0A0", "#606060"]

    sub_color = "#8080b6"
    path = "../plot_img/Fig4b3.pdf"
    violin_monkey(y_data_Patamon, savePath=path, COLOR_SCALE=colors, sub_color=sub_color,
                  ylabel=ylabel, Type=2, layer=2)

    sub_color = "#F6D087"
    path = "../plot_img/Fig4a3.pdf"
    violin_monkey(y_data_Omega, savePath=path, COLOR_SCALE=colors, sub_color=sub_color,
                  ylabel=ylabel, Type=1, layer=2)

def plot_5(K=2):
    Dict = {
        "IS1": "$m_1$",
        "IS2": "$m_2$",
        "PG1": "$g_1$",
        "PG2": "$g_2$",
        "PE": "$e$",
        "BN5": "$b$",
    }
    path = "../plot_data/Fig5/"
    filenames = os.listdir(path)
    savePaths = ["../plot_img/Fig5a.pdf", "../plot_img/Fig5b.pdf", "../plot_img/Fig5c.pdf"]
    for i, filename in enumerate(filenames):
        with open(path + filename, "rb") as file:
            result = pickle.load(file)

        G = result["G"]
        names = result["stateNames"]
        names = [Dict[n] for n in names]
        savePath = savePaths[i]
        draw_state(G, names, savePath)

def plot_6():
    path = "../plot_data/Fig6/"
    fileNames = os.listdir(path)
    filePaths = [path + fileName for fileName in fileNames]
    savePaths = ["../plot_img/Fig6a.pdf", "../plot_img/Fig6c.pdf", "../plot_img/Fig6b.pdf"]
    for j, filePath in enumerate(filePaths):
        df = pd.read_csv(filePath)
        nodes = list(set(list(df["source"]) + list(df["target"])))
        nodes.sort()
        temp = [(i, len(nodes[i])) for i in range(len(nodes))]
        temp = sorted(temp, key=lambda x: x[1])
        temp = [t[0] for t in temp]
        nodes = [nodes[t] for t in temp]

        sapce = [180 / len(df) if i == len(nodes) - 1 or nodes[i][:-3] != nodes[i + 1][:-3] else 0 for i in
                 range(len(nodes))]

        link_cmap = [(df["source"].iloc[i], df["target"].iloc[i], df["color"].iloc[i]) for i in range(len(df))]
        matrix = Matrix.parse_fromto_table(df)
        # Determine length based on quantity
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

        legends = [('#69A89C', 'gl'), ('#DD8162', 'lo'), ('#FFD58F', 'ev'), ('#4E5080', 'ap'), ('#4F74A2', 'en'),
                   ('#7FB069', 'sv'), ('#8C6B93', 'st')]
        legend_elements = [Patch(facecolor=le[0], edgecolor=le[0], label=le[1]) for le in legends]

        # Place the legend on the plot
        font_properties = {'family': 'CMU Serif', 'size': 20}
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.9, 1.1), prop=font_properties)
        fig = circos.plotfig(dpi=300, ax=ax)
        plt.savefig(savePaths[j])
        plt.show()

def plot_sup1():
    print("---------- Fig sup fig1 ----------")
    dark_grey = "#4d4d4d"
    Expert = pd.read_pickle("../plot_data/Sup1/Expert.pkl")
    Amateur = pd.read_pickle("../plot_data/Sup1/Novice.pkl")
    Monkey = pd.read_pickle("../plot_data/Sup1/Monkey.pkl")
    first_second_diff_expert = Expert["series"]
    first_second_diff_amateur = Amateur["series"]
    first_second_diff_monkey = Monkey["series"]
    # Histogram; Probability
    bin = np.arange(0, 1.1, 0.1)

    diff_weight_bin_expert, bin_edges = np.histogram(first_second_diff_expert, bin)
    diff_weight_bin_expert = np.random.uniform(np.repeat(bin_edges[:-1], diff_weight_bin_expert),
                                               np.repeat(bin_edges[1:], diff_weight_bin_expert))

    diff_weight_bin_amateur, bin_edges = np.histogram(first_second_diff_amateur, bin)
    diff_weight_bin_amateur = np.random.uniform(np.repeat(bin_edges[:-1], diff_weight_bin_amateur),
                                                np.repeat(bin_edges[1:], diff_weight_bin_amateur))

    diff_weight_bin_monkey, bin_edges = np.histogram(first_second_diff_monkey, bin)
    diff_weight_bin_monkey = np.random.uniform(np.repeat(bin_edges[:-1], diff_weight_bin_monkey),
                                               np.repeat(bin_edges[1:], diff_weight_bin_monkey))
    # Configurations
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    spec = fig.add_gridspec(1, 4)

    font_properties = {'family': 'CMU Serif', 'size': 20}
    # ax = fig.add_subplot(spec[0, :4])
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sns.violinplot(
        data=[diff_weight_bin_expert, diff_weight_bin_amateur, diff_weight_bin_monkey],
        palette=[dark_grey, dark_grey, dark_grey]
    )
    plt.xticks([0.0, 1, 2], ["Expert", "Novice", "Monkey"], fontsize=20, fontname=font_properties["family"])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20, fontname=font_properties[
        "family"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Strategy Difference \n between 1st and 2nd dominating strategy", fontdict=font_properties)
    path = "../plot_img/Sup-fig-comparison.pdf"
    plt.savefig(path)
    plt.show()

def plot_sup2a():
    Results1=pd.read_pickle("../plot_data/Sup2/uni.pkl")
    Results2 = pd.read_pickle("../plot_data/Sup2/bi.pkl")
    Results3 = pd.read_pickle("../plot_data/Sup2/tri.pkl")
    accuracies = []
    stds = []
    for n in list(Results1.keys()):
        results1 = Results1[n]
        results2 = Results2[n]
        results3 = Results3[n]

        predict_pros = [results1["predict_pro"], results2["predict_pro"], results3["predict_pro"]]
        predict_sets = [results1["predict_sets"], results2["predict_sets"], results3["predict_sets"]]

        accuracy = []
        std = []
        for i in range(len(predict_pros)):
            predict_pro = predict_pros[i]
            predict_set = predict_sets[i]
            number = []
            for j in range(len(predict_pro)):
                sets = predict_set[j]
                pro = predict_pro[j]
                num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for k in range(len(sets)):
                    num[len(sets[k]) - 1] += pro[k]
                index = np.argmax(num)
                if index == i:
                    number.append(1)
                else:
                    number.append(0)
            accuracy.append(np.mean(number))
            std.append(np.std(number))
        accuracies.append(accuracy)
        stds.append(std)
    accuracies = np.array(accuracies).T
    stds = np.array(stds).T

    font_properties = {'family': 'CMU Serif', 'size': 40}
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    colors = ["#B4585F", "#80B6B1", "#F6D087"]
    grammarSet = ["\mathcal{G}_1", "\mathcal{G}_2", "\mathcal{G}_3"]
    index = np.array(list(Results1.keys()))

    for i in range(len(accuracies)):
        plt.plot(index, accuracies[i, :], color=colors[i], label="$" + grammarSet[i] + "$")
        plt.fill_between(index, accuracies[i, :] - stds[i, :],
                         accuracies[i, :] + stds[i, :],
                         color=colors[i], alpha=0.1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(prop=font_properties)
    plt.xlabel("Sample Size", fontdict=font_properties)
    plt.ylabel("Accuracy", fontdict=font_properties)
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.xticks([200, 400, 600, 800, 1000], fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.tight_layout()
    plt.savefig("../plot_img/FigSup2a.pdf")
    plt.show()

def plot_sup2b():
    results1 = pd.read_pickle("../plot_data/Sup2/uni.pkl")
    results2 = pd.read_pickle("../plot_data/Sup2/bi.pkl")
    results3 = pd.read_pickle("../plot_data/Sup2/tri.pkl")
    results = [results1, results2, results3]
    complexity_mean = []
    complexity_std = []
    for i, result in enumerate(results):
        complexity_mean.append([])
        complexity_std.append([])
        num_samples = list(result.keys())
        for j, num_sample in enumerate(num_samples):
            data = result[num_sample]
            coverSet = data["coverSet"]
            complexity = [np.mean([len(c) for c in cover]) for cover in coverSet]
            complexity_mean[i].append(np.mean(complexity))
            complexity_std[i].append(np.std(complexity))
    complexity_mean = np.array(complexity_mean)
    complexity_std = np.array(complexity_std)
    print(complexity_mean)
    print(complexity_std)
    colors = ["#B4585F", "#80B6B1", "#F6D087"]
    num_samples = list(results3.keys())
    legend = []
    grammarSet = ["\mathcal{G}_1", "\mathcal{G}_2", "\mathcal{G}_3"]
    font_properties = {'family': 'CMU Serif', 'size': 40}
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    for i in range(len(grammarSet)):
        plt.plot(num_samples, complexity_mean[i], label="$" + grammarSet[i] + "$", color=colors[i])
        plt.fill_between(num_samples, complexity_mean[i] - complexity_std[i], complexity_mean[i] + complexity_std[i],
                         color=colors[i], alpha=0.1)
        legend.append("$" + grammarSet[i] + "$")
    plt.legend(prop=font_properties)
    plt.hlines(1, num_samples[0], num_samples[-1], linestyles='dashed', colors="black")
    plt.hlines(2, num_samples[0], num_samples[-1], linestyles='dashed', colors="black")
    plt.hlines(3, num_samples[0], num_samples[-1], linestyles='dashed', colors="black")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel("Sample Size", fontdict=font_properties)
    plt.ylabel("LoPS Complexity", fontdict=font_properties)
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.xticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    plt.tight_layout()
    plt.savefig("../plot_img/FigSup2b.pdf")
    plt.show()

if __name__ == '__main__':
    plot_2b()
    plot_2c()
    plot_3a()
    plot_3b()
    plot_4a2_4b2()
    plot_4a3_4b3()
    plot_4c2()
    plot_4c3()
    plot_5()
    plot_6()
    plot_sup1()
    plot_sup2a()
    plot_sup2b()