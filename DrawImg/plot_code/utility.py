import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def significant(data):
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for combination in combinations:
        data1 = data[combination[0] - 1]
        data2 = data[combination[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        # if p < 0.05:
        significant_combinations.append([combination, p])
    return significant_combinations


def violin(y_data, savePath, COLOR_SCALE, means, sub_color, layer=1):
    plt.rcParams['axes.unicode_minus'] = False
    font_properties = {'family': 'CMU Serif', 'size': 40}

    POSITIONS = [0.8 * i for i in range(len(y_data))]
    x_data = [np.array([POSITIONS[i]] * len(m)) for i, m in enumerate(y_data)]
    jitter = 0.04
    x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    # Colors
    BLACK = "#282724"

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Add jittered dots ----------------------------------------------
    for x, y, color in zip(x_jittered, y_data, COLOR_SCALE):
        ax.scatter(x, y, s=100, color=color, alpha=1)
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
    for i, pc in enumerate(violins["bodies"]):
        pc.set_facecolor("none")
        pc.set_edgecolor(COLOR_SCALE[i])
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    # Add mean value labels ------------------------------------------
    for i in range(len(means[0])):
        ax.plot([POSITIONS[0], POSITIONS[1]], [means[0][i], means[1][i]], markerfacecolor=sub_color, marker='o',
                color=sub_color, zorder=3, markersize=15, linestyle="-", lw=2)

    if layer == 1:
        plt.xticks([], fontsize=font_properties["size"],
                   fontname=font_properties["family"])
    else:
        plt.xticks([], fontsize=font_properties["size"],
                   fontname=font_properties["family"])
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath)
    plt.show()


def violin_monkey(y_data, savePath, COLOR_SCALE, sub_color, ylabel, Type=1, layer=1):
    plt.rcParams['axes.unicode_minus'] = False
    font_properties = {'family': 'CMU Serif', 'size': 40}
    POSITIONS = [0.8 * i for i in range(len(y_data))]
    x_data = [np.array([POSITIONS[i]] * len(m)) for i, m in enumerate(y_data)]
    jitter = 0.04
    x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Add jittered dots ----------------------------------------------
    for x, y, color in zip(x_jittered, y_data, COLOR_SCALE):
        ax.scatter(x, y, s=100, color=color, alpha=1)
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
    for i, pc in enumerate(violins["bodies"]):
        pc.set_facecolor("none")
        pc.set_edgecolor(COLOR_SCALE[i])
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    # Add mean value labels ------------------------------------------
    means = [y.mean() for y in y_data]
    ax.plot([POSITIONS[0], POSITIONS[1], POSITIONS[2]], means, markerfacecolor=sub_color, marker='o',
            color=sub_color, zorder=3, markersize=30, linestyle="-", lw=8)

    if layer == 1:
        plt.xticks([], fontsize=font_properties["size"],
                   fontname=font_properties["family"])
    else:
        plt.xticks([], fontsize=font_properties["size"],
                   fontname=font_properties["family"])
    # plt.ylim(yticks[0] - scale, yticks[-1] + scale)
    plt.yticks(fontsize=font_properties["size"],
               fontname=font_properties["family"])
    if Type == 1:
        plt.ylabel(ylabel, fontdict=font_properties)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    # path = "../../result/Fig-gram-depth.pdf"
    if savePath is not None:
        plt.savefig(savePath)
    plt.show()


def draw_state(G, names, save_path):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.rcParams["font.sans-serif"] = "CMU Serif"
    grah = G
    G = nx.Graph()
    labels = {}
    for i in range(len(grah)):
        G.add_node(i)
        labels.update({i: names[i]})
    edgs = []
    for i in range(grah.shape[0]):
        for j in range(grah.shape[1]):
            if grah[i][j] != 0:
                G.add_edge(i, j)
                edgs.append((i, j))
    nx.draw(G, node_size=1500, font_size=10, edge_color="black", labels=labels, node_color=["#CD9710"] * len(grah))
    # nx.draw(G)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

