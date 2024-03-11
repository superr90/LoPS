import os

import numpy as np
import pandas as pd
import pickle
import math
import warnings
import seaborn as sns
import re
from copy import deepcopy

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def ComplexityBysampleNumber():
    with open("../data/simulation/simResult-one--.pkl", "rb") as file:
        results1 = pickle.load(file)

    with open("../data/simulation/simResult-bi--.pkl", "rb") as file:
        results2 = pickle.load(file)

    with open("../data/simulation/simResult-tri--.pkl", "rb") as file:
        results3 = pickle.load(file)

    results = [results1, results2, results3]
    # results = [results3]
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
    # fig = plt.figure(figsize=(15, 12), dpi=300)
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    fmts = ['-o', '-s', '-^']
    for i in range(len(grammarSet)):
        # plt.errorbar(num_samples, complexity_mean[i], yerr=complexity_std[i], label="$" + grammarSet[i] + "$",
        #              fmt=fmts[i], color=colors[i], )
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
    plt.savefig("../result/SupFig-sim-LoPS-Complexity.pdf")
    plt.show()

def AccuracyBysampleNumber():
    with open("../data/simulation/simResult-one--.pkl", "rb") as file:
        Results1 = pickle.load(file)

    with open("../data/simulation/simResult-bi--.pkl", "rb") as file:
        Results2 = pickle.load(file)

    with open("../data/simulation/simResult-tri--.pkl", "rb") as file:
        Results3 = pickle.load(file)

    accuracies = []
    stds = []

    matchs = []
    for n in list(Results1.keys()):
        results1 = Results1[n]
        results2 = Results2[n]
        results3 = Results3[n]

        predict_pros = [results1["predict_pro"], results2["predict_pro"], results3["predict_pro"]]
        predict_sets = [results1["predict_sets"], results2["predict_sets"], results3["predict_sets"]]
        real_sets = []

        accuracy = []
        std = []
        math = []
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
            # number = np.array(number) / len(predict_pro)
            accuracy.append(np.mean(number))
            std.append(np.std(number))
        accuracies.append(accuracy)
        stds.append(std)
    accuracies = np.array(accuracies).T
    stds = np.array(stds).T

    print(accuracies)
    print(stds)
    font_properties = {'family': 'CMU Serif', 'size': 40}
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    colors = ["#B4585F", "#80B6B1", "#F6D087"]
    grammarSet = ["\mathcal{G}_1", "\mathcal{G}_2", "\mathcal{G}_3"]
    index = np.array(list(Results1.keys()))

    fmts = ['-o', '-s', '-^']
    alphas = [1.0, 0.7, 0.4]
    offsets = [-5, 0, 5]
    # for i in range(len(accuracies)):
    #     plt.errorbar(index + offsets[i], accuracies[i, :], yerr=stds[i, :], label="$" + grammarSet[i] + "$",
    #                  fmt=fmts[i], color=colors[i], alpha=alphas[i])

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
    plt.savefig("../result/SupFig-sim-LoPS-accuracy.pdf")
    plt.show()

if __name__ == '__main__':
    ComplexityBysampleNumber()
    AccuracyBysampleNumber()
