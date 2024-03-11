import os
import pandas as pd
import numpy as np
from copy import deepcopy
from utility import name_dict
import scipy.stats as stats

Novice = [0, 9, 14, 16, 24, 26, 32]
ExpertIndex = list(set(list(range(34))) - set(Novice))
ExpertIndex.sort()


def Fig2b():
    path = "../../HumanData/GrammarData/session2/"
    fileNames = os.listdir(path)
    paths = [path + fileName for fileName in fileNames]

    features = []
    for i, path in enumerate(paths):
        print(path)
        data = pd.read_pickle(path)
        gramLen = np.array(data["gramLen"])
        gramStart = np.array(data["gramStart"])
        index = np.where(gramStart == 1)[0]
        gramLen = gramLen[index]
        feature = [0] * 3
        for l in gramLen:
            l = int(l)
            if l <= 3:
                feature[l - 1] += 1
            else:
                feature[2] += 1
        feature = np.array(feature) / np.sum(feature)
        features.append(deepcopy(feature))
    features = np.array(features)

    path = "../../MonkeyData/GrammarData/Year3/"
    fileNames = os.listdir(path)
    paths = [path + fileName for fileName in fileNames]

    monkeyFeature = [[0] * 3, [0] * 3]
    for path in paths:
        print(path)
        data = pd.read_pickle(path)
        gramLen = np.array(data["gramLen"])
        gramStart = np.array(data["gramStart"])
        index = np.where(gramStart == 1)[0]
        gramLen = gramLen[index]

        monkeyNumber = 0 if ('omega' in data["file"].iloc[0].split("-")[2].lower()) else 1
        for l in gramLen:
            if l <= 3:
                monkeyFeature[monkeyNumber][l - 1] += 1
            else:
                monkeyFeature[monkeyNumber][2] += 1
    monkeyFeature = [np.array(monkeyFeature[0]) / np.sum(monkeyFeature[0]),
                     np.array(monkeyFeature[1]) / np.sum(monkeyFeature[1])]

    monkeyFeature = np.array(monkeyFeature)

    features = np.vstack((features, monkeyFeature))

    lables = np.array([0] * len(features))
    for i in Novice:
        if i < len(lables):
            lables[i] = 1

    lables[-2] = 2
    lables[-1] = 3

    data = pd.DataFrame(features)
    data.columns = ['ratio of uni-gram', 'ratio of bi-gram', 'ratio of tri-gram and skip-gram']
    data["labels"] = lables
    colors = ["#B4585F", "#80B6B1", "#F6D087", "#8080b6"]
    legends = ["Expert", "Amateur", "Monkey O", "Monkey P"]
    colors = [colors[i] for i in lables]
    legends = [legends[i] for i in lables]
    data["color"] = colors
    data["legend"] = legends
    data.to_csv("../plot_data/Fig2b.csv")


def Fig2c():
    meanDepthPerGameExpert = []
    meanDepthPerGameNovice = []
    names = []

    path = "../../HumanData/LoPSComplexity/session2/"
    fileNames = os.listdir(path)
    fileNames.sort()
    for k in range(len(fileNames)):
        LoPSComplexity = pd.read_pickle(path + fileNames[k])
        if k in Novice:
            meanDepthPerGameNovice += LoPSComplexity["meanDepthPerGame"]
        else:
            meanDepthPerGameExpert += LoPSComplexity["meanDepthPerGame"]
            names.append(fileNames[k])
    meanDepthPerGameMonkeyO = pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Omega.pkl")["meanDepthPerGame"]
    meanDepthPerGameMonkeyP = pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Patamon.pkl")["meanDepthPerGame"]

    meanDepthPerGame = [meanDepthPerGameExpert, meanDepthPerGameNovice, meanDepthPerGameMonkeyO,
                        meanDepthPerGameMonkeyP]

    meanDepthPerGame = [np.array(m) for m in meanDepthPerGame]

    y_data = deepcopy((meanDepthPerGame))

    data = {
        "Expert": y_data[0],
        "Novice": y_data[1],
        "Monkey O": y_data[2],
        "Monkey P": y_data[3],
    }
    pd.to_pickle(data, "../plot_data/Fig2c.pkl")


def Fig3a():
    reactionTimeHuman = pd.read_pickle("../../HumanData/Performance/session2/reactionTimeHuman.pkl")
    reactionTimeHuman = reactionTimeHuman["meanReactionTime"]
    reactionTimeExpert = []
    reactionTimeAmateur = []
    for i in range(len(reactionTimeHuman)):
        if i in Novice:
            reactionTimeAmateur.append(reactionTimeHuman[i])
        else:
            reactionTimeExpert.append(reactionTimeHuman[i])

    LoPSComplexityExpert = []
    LoPSComplexityAmateur = []
    LoPSComplexityPath = "../../HumanData/LoPSComplexity/session2/"
    fileNames = os.listdir(LoPSComplexityPath)
    for i in range(len(fileNames)):
        complexity = pd.read_pickle(LoPSComplexityPath + fileNames[i])["meanDepthPerSub"]
        if i in Novice:
            LoPSComplexityAmateur.append(complexity)
        else:
            LoPSComplexityExpert.append(complexity)

    reactionTimeMonkeyO = [
        pd.read_pickle("../../MonkeyData/Performance/Year3/reactionTimeMonkeyO.pkl")["meanReactionTime"]]
    reactionTimeMonkeyP = [
        pd.read_pickle("../../MonkeyData/Performance/Year3/reactionTimeMonkeyP.pkl")["meanReactionTime"]]

    LoPSComplexityMonkeyO = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Omega.pkl")["meanDepthPerSub"]]
    LoPSComplexityMonkeyP = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Patamon.pkl")["meanDepthPerSub"]]

    reactionTimeExpert = np.array(reactionTimeExpert) / 120 * 1000
    reactionTimeAmateur = np.array(reactionTimeAmateur) / 120 * 1000
    reactionTimeMonkeyO = np.array(reactionTimeMonkeyO) / 60 * 1000
    reactionTimeMonkeyP = np.array(reactionTimeMonkeyP) / 60 * 1000

    plot_data = {
        "ExpertGramDepth": LoPSComplexityExpert,
        "ExpertReactionTime": reactionTimeExpert,
        "AmateurGramDepth": LoPSComplexityAmateur,
        "AmateurReactionTime": reactionTimeAmateur,
        "MonkeyGramDepth O": LoPSComplexityMonkeyO,
        "MonkeyReactionTime O": reactionTimeMonkeyO,
        "MonkeyGramDepth P": LoPSComplexityMonkeyP,
        "MonkeyReactionTime P": reactionTimeMonkeyP,
    }
    pd.to_pickle(plot_data, "../plot_data/Fig3a.pkl")


def Fig3b():
    rewardHuman = pd.read_pickle("../../HumanData/Performance/session2/rewardHuman.pkl")["rewardPerGame"]
    rewardHuman = [np.mean(rewardHuman[i]) for i in range(len(rewardHuman))]
    rewardTimeExpert = []
    rewardTimeAmateur = []
    for i in range(len(rewardHuman)):
        if i in Novice:
            rewardTimeAmateur.append(rewardHuman[i])
        else:
            rewardTimeExpert.append(rewardHuman[i])

    LoPSComplexityExpert = []
    LoPSComplexityAmateur = []
    LoPSComplexityPath = "../../HumanData/LoPSComplexity/session2/"
    fileNames = os.listdir(LoPSComplexityPath)
    for i in range(len(fileNames)):
        complexity = pd.read_pickle(LoPSComplexityPath + fileNames[i])["meanDepthPerSub"]
        if i in Novice:
            LoPSComplexityAmateur.append(complexity)
        else:
            LoPSComplexityExpert.append(complexity)

    rewardTimeMonkeyO = [pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyO.pkl")["rewardPerGame"]]
    rewardTimeMonkeyP = [pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyP.pkl")["rewardPerGame"]]

    LoPSComplexityMonkeyO = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Omega.pkl")["meanDepthPerSub"]]
    LoPSComplexityMonkeyP = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Patamon.pkl")["meanDepthPerSub"]]

    rewardTimeExpert = np.array(rewardTimeExpert)
    rewardTimeAmateur = np.array(rewardTimeAmateur)
    rewardTimeMonkeyO = np.array([np.mean(rewardTimeMonkeyO)])
    rewardTimeMonkeyP = np.array([np.mean(rewardTimeMonkeyP)])

    plot_data = {
        "ExpertGramDepth": LoPSComplexityExpert,
        "ExpertReward": rewardTimeExpert,
        "AmateurGramDepth": LoPSComplexityAmateur,
        "AmateurReward": rewardTimeAmateur,
        "MonkeyGramDepth O": LoPSComplexityMonkeyO,
        "MonkeyReward O": rewardTimeMonkeyO,
        "MonkeyGramDepth P": LoPSComplexityMonkeyP,
        "MonkeyReward P": rewardTimeMonkeyP,
    }
    pd.to_pickle(plot_data, "../plot_data/Fig3b.pkl")


def Fig4a2_4b2():
    dates = ["Year1", "Year2", "Year3"]
    meanDepthPerGameOmega = []
    meanDepthPerGamePatamon = []
    for date in dates:
        path = "../../MonkeyData/LoPSComplexity/" + date + "/Omega.pkl"
        gramDepth = pd.read_pickle(path)
        meanDepthPerGameMonkey = gramDepth["meanDepthPerGame"]
        meanDepthPerGameOmega.append(deepcopy(meanDepthPerGameMonkey))

        path = "../../MonkeyData/LoPSComplexity/" + date + "/Patamon.pkl"
        gramDepth = pd.read_pickle(path)
        meanDepthPerGameMonkey = gramDepth["meanDepthPerGame"]
        meanDepthPerGamePatamon.append(deepcopy(meanDepthPerGameMonkey))

    meanDepthPerGameOmega = [np.array(m) for m in meanDepthPerGameOmega]
    meanDepthPerGamePatamon = [np.array(m) for m in meanDepthPerGamePatamon]

    y_data_Omega = deepcopy(meanDepthPerGameOmega)
    y_data_Patamon = deepcopy(meanDepthPerGamePatamon)

    plot_data = {
        "y_data_Patamon": y_data_Patamon,
        "y_data_Omega": y_data_Omega,

    }
    pd.to_pickle(plot_data, "../plot_data/Fig4a2-4b2.pkl")


def Fig4a3_4b3():
    dates = ["Year1", "Year2", "Year3"]
    rewardPerGameOmega = []
    rewardPerGamePatamon = []
    for date in dates:
        path = "../../MonkeyData/Performance/" + date + "/rewardMonkeyO.pkl"
        reward = pd.read_pickle(path)
        rewardPerGame = reward["rewardPerGame"]
        rewardPerGameOmega.append(deepcopy(rewardPerGame))

        path = "../../MonkeyData/Performance/" + date + "/rewardMonkeyP.pkl"
        reward = pd.read_pickle(path)
        rewardPerGame = reward["rewardPerGame"]
        rewardPerGamePatamon.append(deepcopy(rewardPerGame))

    rewardPerGameOmega = [np.array(m) for m in rewardPerGameOmega]
    rewardPerGamePatamon = [np.array(m) for m in rewardPerGamePatamon]

    y_data_Omega = deepcopy(rewardPerGameOmega)
    y_data_Patamon = deepcopy(rewardPerGamePatamon)

    plot_data = {
        "y_data_Patamon": y_data_Patamon,
        "y_data_Omega": y_data_Omega,

    }
    pd.to_pickle(plot_data, "../plot_data/Fig4a3-4b3.pkl")


def Fig4c2():
    dict_1_2, dict_2_1 = name_dict()
    path_session1 = "../../HumanData/LoPSComplexity/session1/"
    fileNames_session1 = os.listdir(path_session1)
    complexity1 = {}
    for i in range(len(fileNames_session1)):
        data = pd.read_pickle(path_session1 + fileNames_session1[i])
        allDepth = data["allDepth"][0]
        meanDepth = data["meanDepthPerhuman3"][0]
        meanDepthPerGame = data["meanDepthPerGame3"]

        path = "../../HumanData/GrammarFinall/session1/" + fileNames_session1[i]
        grammar = pd.read_pickle(path)["sets"]
        complexity1.update({fileNames_session1[i][:-4]: [allDepth, meanDepth, grammar, meanDepthPerGame]})

    path = "../../HumanData/FrameData/session2/"
    path_session2 = "../../HumanData/LoPSComplexity/session2/"
    fileNames_session2 = os.listdir(path_session2)
    complexity2 = {}
    for i in range(len(fileNames_session2)):
        data = pd.read_pickle(path_session2 + fileNames_session2[i])
        allDepth = data["allDepth"]
        meanDepth = data["meanDepthPerSub"]
        meanDepthPerGame = data["meanDepthPerGame"]

        path = "../../HumanData/GrammarFinall/session2/" + fileNames_session2[i]
        grammar = pd.read_pickle(path)["sets"]
        complexity2.update({fileNames_session2[i][:-4]: [allDepth, meanDepth, grammar, meanDepthPerGame]})

    c1 = []
    c2 = []
    names = []
    for key1 in complexity1.keys():
        if key1 not in dict_1_2.keys():
            continue
        key2 = dict_1_2[key1]
        if key2 not in complexity2.keys():
            continue
        c1.append(complexity1[key1])
        c2.append(complexity2[key2])
        names.append((key1, key2))
        # print(key1, key2, complexity1[key1][2], complexity2[key2][2])
    diffenence = []
    significant_c1 = [[], [], []]
    significant_c2 = [[], [], []]
    name = []
    for i in range(len(c1)):
        t_statistic, p_value = stats.ttest_ind(c1[i][0], c2[i][0])
        if p_value > 0.05:
            significant_c1[2].append(c1[i][3])
            significant_c2[2].append(c2[i][3])
            continue
        if c2[i][1] > c1[i][1]:
            significant_c1[0].append([c1[i][3], c1[i][1]])
            significant_c2[0].append([c2[i][3], c2[i][1]])
            name.append(names[i][1])
        else:
            significant_c1[1].append([c1[i][3], c1[i][1]])
            significant_c2[1].append([c2[i][3], c2[i][1]])
    print(name)
    gramdepth1 = sum([s[0] for s in significant_c1[0]], [])
    gramdepth2 = sum([s[0] for s in significant_c2[0]], [])
    meanGramDepth1 = [s[1] for s in significant_c1[0]]
    meanGramDepth2 = [s[1] for s in significant_c2[0]]
    y_data = [np.array(gramdepth1), np.array(gramdepth2)]
    means = [meanGramDepth1, meanGramDepth2]

    plot_data = {
        "session1_data": y_data[0],
        "session2_data": y_data[1],
        "means": means,
    }
    pd.to_pickle(plot_data, "../plot_data/Fig4a2.pkl")


def Fig4c3():
    dict_1_2, dict_2_1 = name_dict()
    path1 = "../../HumanData/Performance/session1/rewardHuman.pkl"
    result1 = pd.read_pickle(path1)
    reward1 = result1["rewardPerGame"]
    fileNames1 = result1["fileNames"]
    fileNames1 = ["".join(filename.split("-data")[:1]) for filename in fileNames1]

    path2 = "../../HumanData/Performance/session2/rewardHuman.pkl"
    result2 = pd.read_pickle(path2)
    reward2 = result2["rewardPerGame"]
    fileNames2 = result2["fileNames"]
    fileNames2 = ["".join(filename.split("-data")[:1]) for filename in fileNames2]

    c1 = []
    c2 = []
    names = []
    diffenence = []
    for i in range(len(fileNames1)):
        f1 = fileNames1[i]
        if f1 not in dict_1_2.keys():
            continue
        f2 = dict_1_2[f1]
        if f2 not in fileNames2:
            continue
        c1.append(reward1[i])

        index = fileNames2.index(f2)
        c2.append(reward2[index])
        names.append((f1, f2))

    selected_names = ['231122-402', '131122-402', '051122-402', '141222-402', '091122-401', '161122-404', '151122-401',
                      '241122-402']

    m1 = []
    m2 = []
    s1 = []
    s2 = []

    s11 = []
    s22 = []
    m11 = []
    m22 = []
    for i in range(len(names)):
        if names[i][1] in selected_names:
            t_statistic, p_value = stats.ttest_ind(c1[i], c2[i])
            print(i, p_value)
            m1.append(np.mean(c1[i]))
            s1 += c1[i]
            m2.append(np.mean(c2[i]))
            s2 += c2[i]
        else:
            m11.append(np.mean(c1[i]))
            m22.append(np.mean(c2[i]))
            s11 += c1[i]
            s22 += c2[i]

    t_statistic, p_value = stats.ttest_ind(s1, s2)
    print(np.mean(s1), np.mean(s2), p_value)
    means = [m1, m2]
    y_data = [np.array(s1), np.array(s2)]
    colors = ["black", "#80B6B1", "#F6D087"]
    save_path = "../../result/humanLearningReward.pdf"
    yticks = [-40, -20, 0, 20, 40, 60, 80, 100]
    scale = 10

    plot_data = {
        "session1_data": y_data[0],
        "session2_data": y_data[1],
        "means": means,
    }
    pd.to_pickle(plot_data, "../plot_data/Fig4a3.pkl")



if __name__ == '__main__':
    Fig4c3()
