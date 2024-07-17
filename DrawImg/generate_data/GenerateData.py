import os
import pandas as pd
import numpy as np
from copy import deepcopy
from utility import name_dict
import scipy.stats as stats
import itertools
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

Novice = [0, 9, 14, 16, 24, 26, 32]
ExpertIndex = list(set(list(range(34))) - set(Novice))
ExpertIndex.sort()
scaler = StandardScaler()


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

    reactionTimeExpert = np.array(reactionTimeExpert)
    reactionTimeAmateur = np.array(reactionTimeAmateur)
    reactionTimeMonkeyO = np.array(reactionTimeMonkeyO)
    reactionTimeMonkeyP = np.array(reactionTimeMonkeyP)

    LoPSComplexityAmateur2 = LoPSComplexityAmateur[[4]]
    reactionTimeAmateur2 = reactionTimeAmateur[[4]]

    LoPSComplexityAmateur = LoPSComplexityAmateur[[0, 1, 2, 3, 5, 6]]
    reactionTimeAmateur = reactionTimeAmateur[[0, 1, 2, 3, 5, 6]]
    plot_data = {
        "expert_grammar_depth": LoPSComplexityExpert,
        "expert_reaction_times": reactionTimeExpert,
        "novice_grammar_depth": LoPSComplexityAmateur,
        "novice_reaction_times": reactionTimeAmateur,
        "novice_grammar_depth2": LoPSComplexityAmateur2,
        "novice_reaction_times2": reactionTimeAmateur2,

        "omega_grammar_depth": LoPSComplexityMonkeyO,
        "omega_reaction_times": reactionTimeMonkeyO,
        "patamon_grammar_depth": LoPSComplexityMonkeyP,
        "patamon_reaction_times": reactionTimeMonkeyP,
    }
    pd.to_pickle(plot_data, "../plot_data/Fig3a.pkl")


def Fig3b():
    rewardHuman = pd.read_pickle("../../HumanData/Performance/session2/rewardHuman.pkl")["rewardPerGame"]
    rewardHuman = [np.mean(rewardHuman[i]) for i in range(len(rewardHuman))]
    rewardExpert = []
    rewardAmateur = []
    for i in range(len(rewardHuman)):
        if i in Novice:
            rewardAmateur.append(rewardHuman[i])
        else:
            rewardExpert.append(rewardHuman[i])

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

    rewardMonkeyO = [pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyO.pkl")["rewardPerGame"]]
    rewardMonkeyP = [pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyP.pkl")["rewardPerGame"]]

    LoPSComplexityMonkeyO = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Omega.pkl")["meanDepthPerSub"]]
    LoPSComplexityMonkeyP = [pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Patamon.pkl")["meanDepthPerSub"]]

    rewardExpert = np.array(rewardExpert)
    rewardAmateur = np.array(rewardAmateur)
    rewardMonkeyO = np.array([np.mean(rewardMonkeyO)])
    rewardMonkeyP = np.array([np.mean(rewardMonkeyP)])

    plot_data = {
        "expert_grammar_depth": LoPSComplexityExpert,
        "expert_reward": rewardExpert,
        "novice_grammar_depth": LoPSComplexityAmateur,
        "novice_reward": rewardAmateur,
        "omega_grammar_depth": LoPSComplexityMonkeyO,
        "omega_reward": rewardMonkeyO,
        "patamon_grammar_depth": LoPSComplexityMonkeyP,
        "patamon_reward": rewardMonkeyP,
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


# def Fig4c2():
#     dict_1_2, dict_2_1 = name_dict()
#     path_session1 = "../../HumanData/LoPSComplexity/session1/"
#     fileNames_session1 = os.listdir(path_session1)
#     complexity1 = {}
#     for i in range(len(fileNames_session1)):
#         data = pd.read_pickle(path_session1 + fileNames_session1[i])
#         allDepth = data["allDepth"][0]
#         meanDepth = data["meanDepthPerhuman3"][0]
#         meanDepthPerGame = data["meanDepthPerGame3"]
#
#         path = "../../HumanData/GrammarFinall/session1/" + fileNames_session1[i]
#         grammar = pd.read_pickle(path)["sets"]
#         complexity1.update({fileNames_session1[i][:-4]: [allDepth, meanDepth, grammar, meanDepthPerGame]})
#
#     path = "../../HumanData/FrameData/session2/"
#     path_session2 = "../../HumanData/LoPSComplexity/session2/"
#     fileNames_session2 = os.listdir(path_session2)
#     complexity2 = {}
#     for i in range(len(fileNames_session2)):
#         data = pd.read_pickle(path_session2 + fileNames_session2[i])
#         allDepth = data["allDepth"]
#         meanDepth = data["meanDepthPerSub"]
#         meanDepthPerGame = data["meanDepthPerGame"]
#
#         path = "../../HumanData/GrammarFinall/session2/" + fileNames_session2[i]
#         grammar = pd.read_pickle(path)["sets"]
#         complexity2.update({fileNames_session2[i][:-4]: [allDepth, meanDepth, grammar, meanDepthPerGame]})
#
#     c1 = []
#     c2 = []
#     names = []
#     for key1 in complexity1.keys():
#         if key1 not in dict_1_2.keys():
#             continue
#         key2 = dict_1_2[key1]
#         if key2 not in complexity2.keys():
#             continue
#         c1.append(complexity1[key1])
#         c2.append(complexity2[key2])
#         names.append((key1, key2))
#         # print(key1, key2, complexity1[key1][2], complexity2[key2][2])
#     diffenence = []
#     significant_c1 = [[], [], []]
#     significant_c2 = [[], [], []]
#     name = []
#     for i in range(len(c1)):
#         t_statistic, p_value = stats.ttest_ind(c1[i][0], c2[i][0])
#         if p_value > 0.05:
#             significant_c1[2].append(c1[i][3])
#             significant_c2[2].append(c2[i][3])
#             continue
#         if c2[i][1] > c1[i][1]:
#             significant_c1[0].append([c1[i][3], c1[i][1]])
#             significant_c2[0].append([c2[i][3], c2[i][1]])
#             name.append(names[i][1])
#         else:
#             significant_c1[1].append([c1[i][3], c1[i][1]])
#             significant_c2[1].append([c2[i][3], c2[i][1]])
#     print(name)
#     gramdepth1 = sum([s[0] for s in significant_c1[0]], [])
#     gramdepth2 = sum([s[0] for s in significant_c2[0]], [])
#     meanGramDepth1 = [s[1] for s in significant_c1[0]]
#     meanGramDepth2 = [s[1] for s in significant_c2[0]]
#     y_data = [np.array(gramdepth1), np.array(gramdepth2)]
#     means = [meanGramDepth1, meanGramDepth2]
#
#     plot_data = {
#         "session1_data": y_data[0],
#         "session2_data": y_data[1],
#         "means": means,
#     }
#     pd.to_pickle(plot_data, "../plot_data/Fig4a2.pkl")
#
#
# def Fig4c3():
#     dict_1_2, dict_2_1 = name_dict()
#     path1 = "../../HumanData/Performance/session1/rewardHuman.pkl"
#     result1 = pd.read_pickle(path1)
#     reward1 = result1["rewardPerGame"]
#     fileNames1 = result1["fileNames"]
#     fileNames1 = ["".join(filename.split("-data")[:1]) for filename in fileNames1]
#
#     path2 = "../../HumanData/Performance/session2/rewardHuman.pkl"
#     result2 = pd.read_pickle(path2)
#     reward2 = result2["rewardPerGame"]
#     fileNames2 = result2["fileNames"]
#     fileNames2 = ["".join(filename.split("-data")[:1]) for filename in fileNames2]
#
#     c1 = []
#     c2 = []
#     names = []
#     diffenence = []
#     for i in range(len(fileNames1)):
#         f1 = fileNames1[i]
#         if f1 not in dict_1_2.keys():
#             continue
#         f2 = dict_1_2[f1]
#         if f2 not in fileNames2:
#             continue
#         c1.append(reward1[i])
#
#         index = fileNames2.index(f2)
#         c2.append(reward2[index])
#         names.append((f1, f2))
#
#     selected_names = ['231122-402', '131122-402', '051122-402', '141222-402', '091122-401', '161122-404', '151122-401',
#                       '241122-402']
#
#     m1 = []
#     m2 = []
#     s1 = []
#     s2 = []
#
#     s11 = []
#     s22 = []
#     m11 = []
#     m22 = []
#     for i in range(len(names)):
#         if names[i][1] in selected_names:
#             t_statistic, p_value = stats.ttest_ind(c1[i], c2[i])
#             print(i, p_value)
#             m1.append(np.mean(c1[i]))
#             s1 += c1[i]
#             m2.append(np.mean(c2[i]))
#             s2 += c2[i]
#         else:
#             m11.append(np.mean(c1[i]))
#             m22.append(np.mean(c2[i]))
#             s11 += c1[i]
#             s22 += c2[i]
#
#     t_statistic, p_value = stats.ttest_ind(s1, s2)
#     print(np.mean(s1), np.mean(s2), p_value)
#     means = [m1, m2]
#     y_data = [np.array(s1), np.array(s2)]
#     colors = ["black", "#80B6B1", "#F6D087"]
#     save_path = "../../result/humanLearningReward.pdf"
#     yticks = [-40, -20, 0, 20, 40, 60, 80, 100]
#     scale = 10
#
#     plot_data = {
#         "session1_data": y_data[0],
#         "session2_data": y_data[1],
#         "means": means,
#     }
#     pd.to_pickle(plot_data, "../plot_data/Fig4a3.pkl")


def Fig5a():
    expert = ['041122-401', '041122-403', '041222-401', '051122-401', '051122-402', '051122-501', '071122-401',
              '071122-402', '081122-401', '091122-401', '101122-401', '101122-402', '131122-402', '141222-401',
              '141222-402', '151122-401', '161122-401', '161122-402', '161122-403', '161122-404', '180522-502',
              '231122-401', '231122-402', '241122-401', '241122-402', '301122-402', '311022-501', 'cluster00', ]
    novice = ['031222-401', '071122-403', '111122-401', '131222-401', '171122-401', '211122-402', '311022-401']
    dict_1_2, dict_2_1 = name_dict()

    path_session1 = "../../HumanData/LoPSComplexity/session1/"
    fileNames_session1 = os.listdir(path_session1)
    complexity1 = {}
    for i in range(len(fileNames_session1)):
        data = pd.read_pickle(path_session1 + fileNames_session1[i])
        allDepth = data["meanDepthPerGame3"]
        meanDepth = data["meanDepthPerhuman3"][0]
        complexity1.update({fileNames_session1[i][:-4]: [allDepth, meanDepth]})

    path_session2 = "../../HumanData/LoPSComplexity/session2/"
    fileNames_session2 = os.listdir(path_session2)
    names = os.listdir("../../MyData/fmriFormedData2/")
    complexity2 = {}
    for i in range(len(fileNames_session2)):
        data = pd.read_pickle(path_session2 + fileNames_session2[i])
        allDepth = data["meanDepthPerGame3"]
        meanDepth = data["meanDepthPerhuman3"]
        complexity2.update({names[i][:-4]: [allDepth, meanDepth]})

    names1 = []
    names2 = []
    complexity_difference = []
    complexity_session1 = []
    complexity_session2 = []
    for key1 in complexity1.keys():
        if key1 not in dict_1_2.keys():
            continue
        key2 = dict_1_2[key1]
        if key2 not in complexity2.keys():
            continue
        names1.append(key1)
        names2.append(key2)
        cd = complexity2[key2][1] - complexity1[key1][1]
        complexity_difference.append(cd)
        complexity_session1.append(complexity1[key1][1])
        complexity_session2.append(complexity2[key2][1])

    complexity_expert = [[], []]
    complexity_novice = [[], []]
    for i in range(len(complexity_difference)):
        if names2[i] in expert:
            complexity_expert[0].append(complexity_session1[i])
            complexity_expert[1].append(complexity_session2[i])
        else:
            complexity_novice[0].append(complexity_session1[i])
            complexity_novice[1].append(complexity_session2[i])
    save_data = {
        "complexity_expert": complexity_expert,
        "complexity_novice": complexity_novice,
    }
    pd.to_pickle(save_data, "../plot_data/Fig5a.pkl")


def Fig5b():
    expert = ['041122-401', '041122-403', '041222-401', '051122-401', '051122-402', '051122-501', '071122-401',
              '071122-402', '081122-401', '091122-401', '101122-401', '101122-402', '131122-402', '141222-401',
              '141222-402', '151122-401', '161122-401', '161122-402', '161122-403', '161122-404', '180522-502',
              '231122-401', '231122-402', '241122-401', '241122-402', '301122-402', '311022-501', 'cluster00', ]
    dict_1_2, dict_2_1 = name_dict()

    path_session1 = "../../HumanData/LoPSComplexity/session1/"
    fileNames_session1 = os.listdir(path_session1)
    complexity1 = {}
    for i in range(len(fileNames_session1)):
        data = pd.read_pickle(path_session1 + fileNames_session1[i])
        allDepth = data["meanDepthPerGame3"]
        meanDepth = data["meanDepthPerhuman3"][0]
        complexity1.update({fileNames_session1[i][:-4]: [allDepth, meanDepth]})

    path_session2 = "../../HumanData/LoPSComplexity/session2/"
    fileNames_session2 = os.listdir(path_session2)
    names = os.listdir(path_session2)
    complexity2 = {}
    for i in range(len(fileNames_session2)):
        data = pd.read_pickle(path_session2 + fileNames_session2[i])
        allDepth = data["meanDepthPerGame3"]
        meanDepth = data["meanDepthPerhuman3"]
        complexity2.update({names[i][:-4]: [allDepth, meanDepth]})

    names1 = []
    names2 = []
    complexity_difference = []
    for key1 in complexity1.keys():
        if key1 not in dict_1_2.keys():
            continue
        key2 = dict_1_2[key1]
        if key2 not in complexity2.keys():
            continue
        names1.append(key1)
        names2.append(key2)
        cd = complexity2[key2][1] - complexity1[key1][1]
        complexity_difference.append(cd)

    path = "../../HumanData/Performance/session1/reward.pkl"
    reward1 = pd.read_pickle("../../HumanData/Performance/session1/reward.pkl")
    reward2 = pd.read_pickle("../../HumanData/Performance/session2/reward.pkl")

    reward_name1 = reward1["fileNames"]
    reward1 = reward1["rewardPerGame"]
    reward_name2 = reward2["fileNames"]
    reward2 = reward2["rewardPerGame"]

    reward_difference = [0] * len(names1)
    for key1 in reward_name1:
        if key1 not in dict_1_2.keys() or key1 not in names1:
            continue
        key2 = dict_1_2[key1]
        if key2 not in reward_name2:
            continue
        index = names1.index(key1)

        index1 = reward_name1.index(key1)
        index2 = reward_name2.index(key2)
        r1 = np.mean(reward1[index1])
        r2 = np.mean(reward2[index2])
        reward_difference[index] = r2 - r1

    selected_names = ['231122-402', '131122-402', '051122-501', '051122-402', '141222-402', '091122-401', '151122-401',
                      '241122-402']

    complexity_novice = []
    complexity_expert = []

    reward_difference_novice = []
    reward_difference_expert = []
    names2_expert = []
    names1_expert = []
    for i in range(len(complexity_difference)):
        if names2[i] not in expert:
            reward_difference_novice.append(reward_difference[i])
            complexity_novice.append(complexity_difference[i])
        else:
            reward_difference_expert.append(reward_difference[i])
            complexity_expert.append(complexity_difference[i])
            names2_expert.append(names2[i])
            names1_expert.append(names1[i])

    save_data = {
        "complexity_difference": complexity_difference,
        "reward_difference": reward_difference,
        "names1": names1,
        "names2": names2,
        "selected_names": selected_names,
        "expert": expert
    }
    pd.to_pickle(save_data, "./Fig5b.pkl")


def FigS3():
    expert = ['041122-401', '041122-403', '041222-401', '051122-401', '051122-402', '051122-501', '071122-401',
              '071122-402', '081122-401', '091122-401', '101122-401', '101122-402', '131122-402', '141222-401',
              '141222-402', '151122-401', '161122-401', '161122-402', '161122-403', '161122-404', '180522-502',
              '231122-401', '231122-402', '241122-401', '241122-402', '301122-402', '311022-501', 'cluster00', ]
    novice = ['031222-401', '071122-403', '111122-401', '131222-401', '171122-401', '211122-402', '311022-401']
    dict_1_2, dict_2_1 = name_dict()

    path_session1 = "../../HumanData/LoPSComplexity/session1/"
    fileNames_session1 = os.listdir(path_session1)
    complexity1 = {}
    for i in range(len(fileNames_session1)):
        data = pd.read_pickle(path_session1 + fileNames_session1[i])
        # allDepth = data["allDepth3"][0]
        allDepth = data["meanDepthPerGame3"]
        # allDepth = data["meanDepthPerhuman3"]
        meanDepth = data["meanDepthPerhuman3"][0]

        complexity1.update({fileNames_session1[i][:-4]: [allDepth, meanDepth]})

    path_session2 = "../../HumanData/LoPSComplexity/session2/"
    fileNames_session2 = os.listdir(path_session2)
    names = os.listdir("../../HumanData/FormedData/session2/")
    complexity2 = {}
    for i in range(len(fileNames_session2)):
        data = pd.read_pickle(path_session2 + fileNames_session2[i])
        allDepth = data["allDepth3"]
        allDepth = data["meanDepthPerGame3"]
        # allDepth = [data["meanDepthPerhuman3"]]
        meanDepth = data["meanDepthPerhuman3"]

        complexity2.update({names[i][:-4]: [allDepth, meanDepth]})

    names1 = []
    names2 = []
    complexity_difference = []
    for key1 in complexity1.keys():
        if key1 not in dict_1_2.keys():
            continue
        key2 = dict_1_2[key1]
        if key2 not in complexity2.keys():
            continue
        names1.append(key1)
        names2.append(key2)
        cd = complexity2[key2][1] - complexity1[key1][1]
        complexity_difference.append(cd)

    reward1 = pd.read_pickle("../../../Monkey_Analysis/eye_data_process/Perfromance/reward.pkl")
    reward2 = pd.read_pickle("../../../Monkey_Analysis/fmri_data_process/Perfromance/reward.pkl")

    reward_name1 = reward1["fileNames"]
    reward1 = reward1["rewardPerGame"]
    reward_name2 = reward2["fileNames"]
    reward2 = reward2["rewardPerGame"]

    reward_difference = [0] * len(names1)
    for key1 in reward_name1:
        if key1 not in dict_1_2.keys() or key1 not in names1:
            continue
        key2 = dict_1_2[key1]
        if key2 not in reward_name2:
            continue
        index = names1.index(key1)

        index1 = reward_name1.index(key1)
        index2 = reward_name2.index(key2)
        r1 = np.mean(reward1[index1])
        r2 = np.mean(reward2[index2])
        reward_difference[index] = r2 - r1

    selected_names = ['231122-402', '131122-402', '051122-501', '051122-402', '141222-402', '091122-401', '151122-401',
                      '241122-402']

    complexity_difference_25 = []
    reward_difference_25 = []
    complexity_difference_8 = []
    reward_difference_8 = []
    complexity_difference_patamon = [0.05376701905037784, 0.13382140376065665]
    complexity_difference_omega = [0.14890671627940866, 0.09990485101087665]

    reward_difference_patamon = [4.63547527013408, 2.164364837582827]
    reward_difference_omega = [15.290336787074043, -2.694005869725615]

    for i in range(len(complexity_difference)):
        if names2[i] not in selected_names:
            complexity_difference_25.append(complexity_difference[i])
            reward_difference_25.append(reward_difference[i])
        else:
            complexity_difference_8.append(complexity_difference[i])
            reward_difference_8.append(reward_difference[i])

    save_data = {
        "complexity_difference": complexity_difference,
        "reward_difference": reward_difference,
        "complexity_difference_omega": complexity_difference_omega,
        "reward_difference_omega": reward_difference_omega,
        "complexity_difference_patamon": complexity_difference_patamon,
        "reward_difference_patamon": reward_difference_patamon,
        "names1": names1,
        "names2": names2,
        "selected_names": selected_names,
        "expert": expert
    }
    pd.to_pickle(save_data, "../plot_data/FigS3.pkl")


def get_LROfS_CNT():
    "linear regression of score~complexity,total number of actions,total time of the game"
    pass
    path = "../../HumanData/LoPSComplexity/session2/"
    grammar_path = "../../HumanData/LoPSComplexity/session2/"
    filenames = os.listdir(grammar_path)
    grammar_depths = []
    for k, filename in enumerate(filenames):
        result = pd.read_pickle(grammar_path + filename)
        depth = result["meanDepthPerhuman3"]
        grammar_depths.append(depth)

    path = "../../MonkeyData/Performance/Year3/rewardMonkeyP.pkl"
    omega_grammar_depth = pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Omega.pkl")[
        "meanDepthPerhuman3"]
    patamon_grammar_depth = pd.read_pickle("../../MonkeyData/LoPSComplexity/Year3/Patamon.pkl")[
        "meanDepthPerhuman3"]
    complexity = grammar_depths + omega_grammar_depth + patamon_grammar_depth

    performance = pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyO.pkl")
    omega_performance = pd.read_pickle("../../MonkeyData/Performance/Year3/rewardMonkeyO.pkl")
    patamon_performance = pd.read_pickle(
        "../../MonkeyData/Performance/Year3/rewardMonkeyP.pkl")
    # feature2 total number of actions
    total_number_of_actions = performance["total_number_of_actions"] + [
        omega_performance["total_number_of_actions"]] + [patamon_performance["total_number_of_actions"]]

    # feature3 total time of the games
    total_time_of_the_games = performance["total_time_of_the_games"] + [
        omega_performance["total_time_of_the_games"]] + [patamon_performance["total_time_of_the_games"]]

    # label
    rewards = [np.mean(reward) for reward in performance["rewardPerGame"]] + [
        np.mean(omega_performance["rewardPerGame"])] + [np.mean(patamon_performance["rewardPerGame"])]

    df = pd.DataFrame({
        "C": complexity,
        "N": total_number_of_actions,
        "T": total_time_of_the_games,
        "R": rewards,
    })

    df.to_pickle("./LROfS_CNT.pkl")


def get_gram_pro(data):
    data = data[data["gramStart"] == 1]
    gram = np.array(data["gram"])
    gramLen = np.array(data["gramLen"])

    indexs = np.where((gram == 'N') & (gramLen > 2))[0]
    skip_len = 0
    if len(indexs) != 0:
        gram[indexs] = "skip"
        skip_len = np.mean(gramLen[indexs] - 1)
    from collections import Counter
    counter = Counter(gram)

    features = {
        "L": 0, "G": 0, "1": 0, "2": 0, "A": 0, "E": 0, "N": 0, "V": 0, "S": 0,
        "LG": 0, "LE": 0, "GL": 0, "AL": 0, "EA": 0, "LEA": 0, "EAL": 0, "EAG": 0, "SEA": 0, "EAGL": 0, "skip": 0
    }
    for key in features:
        if key in counter.keys():
            features[key] = counter[key]
    return features, skip_len


def get_RD_GD():
    name2 = ['311022-501', '211122-402', '231122-402', '041122-401', '311022-401', '071122-403', '041122-403',
             '071122-401',
             '131122-402', '111122-401', '051122-501', '051122-402', '141222-402', '171122-401', '081122-401',
             '131222-401',
             '091122-401', '141222-401', '101122-402', '101122-401', '071122-402', '161122-403', '161122-401',
             '161122-404',
             '231122-401', '161122-402', '151122-401', '241122-402', '180522-502', '241122-401', '301122-402',
             '041222-401',
             '031222-401']
    name1 = ['111111-001', '111111-002', '111111-003', '111111-004', '111111-005', '111111-006', '111111-007',
             '111111-008',
             '111111-010', '111111-012', '111111-013', '111111-014', '111111-015', '111111-017', '111111-018',
             '111111-019',
             '111111-022', '111111-023', '111111-024', '111111-025', '111111-026', '111111-027', '111111-028',
             '111111-029',
             '111111-030', '111111-031', '111111-032', '111111-033', '111111-034', '111111-036', '111111-037',
             '111111-038',
             '111111-039']
    path = "../../HumanData/GrammarData/session1/"
    grammar_path1 = "../../HumanData/GrammarData/session1/"
    grammar_path2 = "../../HumanData/GrammarData/session2/"
    features = []
    for i in range(len(name1[:])):
        print(i)
        # data1 = pd.read_pickle(grammar_path1 + name1[i] + "-gram.pkl")
        data2 = pd.read_pickle(grammar_path2 + name2[i] + ".pkl")
        # grammar1, skip_len1 = get_gram_pro(data1)
        grammar2, skip_len2 = get_gram_pro(data2)

        # num = np.sum(list(grammar1.values()))
        # grammar1_frequency = {key: grammar1[key] / num for key in grammar1.keys()}

        num = np.sum(list(grammar2.values()))
        grammar2_frequency = {key: grammar2[key] / num for key in grammar2.keys()}

        keys = list(grammar2_frequency.keys())
        feature = []
        for key in keys:
            feature.append(grammar2_frequency[key])
        features.append(feature)

    features = np.array(features)
    df = pd.DataFrame(features, columns=keys)

    keys = [k for k in keys if len(k) > 1]
    df = df[keys]

    path1 = "../../HumanData/Performance/session1/rewardHuman.pkl"
    result1 = pd.read_pickle(path1)
    reward_name1 = result1["fileNames"]
    result1 = result1["rewardPerGame"]

    path2 = "../../HumanData/Performance/session2/rewardHuman.pkl"
    result2 = pd.read_pickle(path2)
    reward_name2 = result2["fileNames"]
    result2 = result2["rewardPerGame"]

    reward_difference = []
    for i in range(len(name1[:])):
        index1 = reward_name1.index(name1[i])
        index2 = reward_name2.index(name2[i])
        r1 = np.mean(result1[index1])
        r2 = np.mean(result2[index2])
        reward_difference.append(r2 - r1)
    df["RD"] = reward_difference

    RD_GD = {
        "df": df,
        "keys": keys
    }
    pd.to_pickle(RD_GD, "./RD_GD.pkl")


def static_LROfS_CNT():
    df = pd.read_pickle("../plot_data/LROfS_CNT.pkl")
    features = ['C', 'N', 'T']
    combinations = []
    for i in range(1, len(features) + 1):
        combinations.extend(itertools.combinations(features, i))

    combinations = combinations[-1:]
    x_sticks = []
    results = []
    labels_pair = []
    np.random.seed(123)
    indexs = np.random.randint(0, len(df), size=1000)
    for combo in combinations:
        X_combo = df[list(combo)]
        X_combo = sm.add_constant(X_combo)

        X = X_combo.iloc[indexs].values
        y = df['R'].iloc[indexs].values
        # model = sm.OLS(X_combo, df['R']).fit()
        # print(model.summary())
        from sklearn.utils import resample

        n_iterations = 1000
        n_size = len(X)
        coefficients = []
        R2 = []
        for i in range(n_iterations):
            X_resample, y_resample = resample(X, y, n_samples=n_size, replace=True)
            model = sm.OLS(y_resample, X_resample).fit()
            coefficients.append(model.params)

        coefficients = pd.DataFrame(coefficients, columns=['const', 'C', 'N', 'T'])

        mean_coeff = coefficients.mean()
        conf_int = coefficients.quantile([0.025, 0.975])

        print(mean_coeff)
        print(conf_int)


def static_RD_GD():
    """
    reward difference
    grammar difference
    :return:
    """
    df = pd.read_pickle("../plot_data/RD_GD.pkl")
    columns = ['f', 'c']
    X = df[columns].values
    y = df["R"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_res = np.std(residuals)
    threshold = 2 * std_res
    non_outliers = np.abs(residuals) < threshold
    X_filtered = X[non_outliers]
    y_filtered = y[non_outliers]

    model_filtered = LinearRegression()
    model_filtered.fit(X_filtered, y_filtered)
    y_pred_filtered = model_filtered.predict(X_filtered)

    # 计算去除异常点后的均方误差
    mse_filtered = mean_squared_error(y_filtered, y_pred_filtered)
    print(f'Filtered MSE: {mse_filtered}')

    df = df.iloc[np.where(np.abs(residuals) < threshold)[0]]
    df = df.reset_index(drop=True)
    X_combo = df[columns]
    X_combo = pd.DataFrame(scaler.fit_transform(X_combo), columns=X_combo.columns)
    X_combo = sm.add_constant(X_combo)  # 添加常数项（截距）
    model = sm.OLS(df["R"], X_combo).fit()
    prediction = model.predict(X_combo)
    print(model.summary())


def grammar_contribute_performance_data():
    """
    The contribution of grammar to performance
    :return:
    """
    name2 = ['311022-501', '211122-402', '231122-402', '041122-401', '311022-401', '071122-403', '041122-403',
             '071122-401',
             '131122-402', '111122-401', '051122-501', '051122-402', '141222-402', '171122-401', '081122-401',
             '131222-401',
             '091122-401', '141222-401', '101122-402', '101122-401', '071122-402', '161122-403', '161122-401',
             '161122-404',
             '231122-401', '161122-402', '151122-401', '241122-402', '180522-502', '241122-401', '301122-402',
             '041222-401',
             '031222-401']
    name1 = ['111111-001', '111111-002', '111111-003', '111111-004', '111111-005', '111111-006', '111111-007',
             '111111-008',
             '111111-010', '111111-012', '111111-013', '111111-014', '111111-015', '111111-017', '111111-018',
             '111111-019',
             '111111-022', '111111-023', '111111-024', '111111-025', '111111-026', '111111-027', '111111-028',
             '111111-029',
             '111111-030', '111111-031', '111111-032', '111111-033', '111111-034', '111111-036', '111111-037',
             '111111-038',
             '111111-039']

    grammar_path2 = "../../HumanData/GrammarData/session2/"
    features = []
    for i in range(len(name1[:])):
        data2 = pd.read_pickle(grammar_path2 + name2[i] + ".pkl")
        grammar2, skip_len2 = get_gram_pro(data2)
        num = np.sum(list(grammar2.values()))
        grammar2_frequency = {key: grammar2[key] / num for key in grammar2.keys()}
        keys = list(grammar2_frequency.keys())
        feature = []
        for key in keys:
            feature.append(grammar2_frequency[key])
        features.append(feature)

    features = np.array(features)
    df = pd.DataFrame(features, columns=keys)

    keys = [k for k in keys if len(k) > 1]
    df = df[keys]

    path1 = "../../HumanData/Performance/session1/reward.pkl"
    result1 = pd.read_pickle(path1)
    reward_name1 = result1["fileNames"]
    result1 = result1["rewardPerGame"]

    path2 = "../../HumanData/Performance/session2/reward.pkl"
    result2 = pd.read_pickle(path2)
    reward_name2 = result2["fileNames"]
    result2 = result2["rewardPerGame"]

    reward_difference = []
    for i in range(len(name1[:])):
        index1 = reward_name1.index(name1[i])
        index2 = reward_name2.index(name2[i])
        r1 = np.mean(result1[index1])
        r2 = np.mean(result2[index2])
        reward_difference.append(r2 - r1)
    df["RD"] = reward_difference


def grammar_contribute_performance():
    data = pd.read_pickle("../plot_data/contribution_of_grammar.pkl")
    df = data["df"]
    keys = data["keys"]
    corrs = []
    CI = []
    for key in keys:
        r = np.corrcoef(df[key], df["RD"])[0, 1]
        corrs.append(r)

        X = df[key]
        y = df["RD"]
        from sklearn.utils import resample
        n_iterations = 1000
        n_size = len(X)
        rs = []
        for i in range(n_iterations):
            X_resample, y_resample = resample(X, y, n_samples=n_size, replace=True)
            r = np.corrcoef(X_resample, y_resample)[0, 1]
            rs.append(r)
        coefficients = pd.DataFrame(rs, columns=['r'])
        conf_int = coefficients.quantile([0.025, 0.975])
        lower = conf_int['r'].iloc[0]
        up = conf_int['r'].iloc[1]
        CI.append((lower, up))

    indexs = np.argsort(corrs)[::-1]

    for i in indexs[:5]:
        print(keys[i], corrs[i], CI[i])


if __name__ == '__main__':
    pass
    # grammar_contribute_performance()
    # static_RD_GD()
    # static_LROfS_CNT()
    # FigS3()
    # Fig3a()
    # Fig3b()
