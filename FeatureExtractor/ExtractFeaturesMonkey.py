import os

import pandas as pd
import numpy as np
import pickle as pkl
import sys
import copy
import pickle
import warnings
from functools import partial
import multiprocessing

import warnings

warnings.filterwarnings("ignore")
sys.path.append("./")
# from Inference_v2 import readLocDistance, readAdjacentMap, _adjacentDist,_adjacentBeans, _ghostModeDist, oneHot

inf_val = 100
reborn_pos = (14, 27)


# -----------------------------------------------------------------------------

def _readAdjacentMap(filename):
    '''
    Read in the adjacent info of the map.
    :param filename: File name.
    :return: A dictionary denoting adjacency of the map.
    '''
    adjacent_data = pd.read_csv(filename)

    for each in ['pos', 'left', 'right', 'up', 'down']:
        adjacent_data[each] = adjacent_data[each].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    dict_adjacent_data = {}
    for each in adjacent_data.values:
        dict_adjacent_data[each[1]] = {}
        dict_adjacent_data[each[1]]["left"] = each[2] if not isinstance(each[2], float) else np.nan
        dict_adjacent_data[each[1]]["right"] = each[3] if not isinstance(each[3], float) else np.nan
        dict_adjacent_data[each[1]]["up"] = each[4] if not isinstance(each[4], float) else np.nan
        dict_adjacent_data[each[1]]["down"] = each[5] if not isinstance(each[5], float) else np.nan
    if (-1, 18) not in dict_adjacent_data:
        dict_adjacent_data[(-1, 18)] = {}
    if (29, 18) not in dict_adjacent_data:
        dict_adjacent_data[(29, 18)] = {}
    if (30, 18) not in dict_adjacent_data:
        dict_adjacent_data[(30, 18)] = {}
    dict_adjacent_data[(-1, 18)]["left"] = (30, 18)
    dict_adjacent_data[(-1, 18)]["right"] = (0, 18)
    dict_adjacent_data[(-1, 18)]["up"] = np.nan
    dict_adjacent_data[(-1, 18)]["down"] = np.nan
    dict_adjacent_data[(29, 18)]["left"] = (28, 18)
    dict_adjacent_data[(29, 18)]["right"] = (30, 18)
    dict_adjacent_data[(29, 18)]["up"] = np.nan
    dict_adjacent_data[(29, 18)]["down"] = np.nan
    dict_adjacent_data[(30, 18)]["left"] = (29, 18)
    dict_adjacent_data[(30, 18)]["right"] = (-1, 18)
    dict_adjacent_data[(30, 18)]["up"] = np.nan
    dict_adjacent_data[(30, 18)]["down"] = np.nan
    return dict_adjacent_data


def _readLocDistance(filename):
    '''
    Read in the location distance.
    :param filename: File name.
    :return: A pandas.DataFrame denoting the dijkstra distance between every two locations of the map.
    '''
    locs_df = pd.read_csv(filename)[["pos1", "pos2", "dis"]]
    locs_df.pos1, locs_df.pos2 = (
        locs_df.pos1.apply(eval),
        locs_df.pos2.apply(eval)
    )
    dict_locs_df = {}
    for each in locs_df.values:
        if each[0] not in dict_locs_df:
            dict_locs_df[each[0]] = {}
        dict_locs_df[each[0]][each[1]] = each[2]
    # correct the distance between two ends of the tunnel
    if (-1, 18) not in dict_locs_df:
        dict_locs_df[(-1, 18)] = {}
    if (29, 18) not in dict_locs_df:
        dict_locs_df[(29, 18)] = {}
    if (30, 18) not in dict_locs_df:
        dict_locs_df[(30, 18)] = {}
    for val in dict_locs_df[(0, 18)]:
        dict_locs_df[(-1, 18)][val] = dict_locs_df[(0, 18)][val] - 1 if val[0] >= 15 else dict_locs_df[(0, 18)][val] + 1
    for val in dict_locs_df[(29, 18)]:
        dict_locs_df[(30, 18)][val] = dict_locs_df[(29, 18)][val] + 1 if val[0] >= 15 else dict_locs_df[(29, 18)][
                                                                                               val] - 1
    dict_locs_df[(-1, 18)][(30, 18)] = 1
    dict_locs_df[(-1, 18)][(0, 18)] = 1
    dict_locs_df[(0, 18)][(1, 18)] = 1
    dict_locs_df[(1, 18)][(0, 18)] = 1
    dict_locs_df[(30, 18)][(29, 18)] = 1
    dict_locs_df[(30, 18)][(-1, 18)] = 1
    dict_locs_df[(29, 18)][(30, 18)] = 1
    dict_locs_df[(29, 18)][(28, 18)] = 1
    return dict_locs_df


def _adjacentBeans(pacmanPos, beans, type, locs_df):
    # Pacman in tunnel
    if pacmanPos == (30, 18):
        pacmanPos = (29, 18)
    if pacmanPos == (0, 18):
        pacmanPos = (1, 18)
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent beans num
    if adjacent not in locs_df:
        bean_num = 0
    else:
        bean_num = (
            0 if isinstance(beans, float) else len(np.where(
                np.array([0 if adjacent == each else locs_df[adjacent][each] for each in beans]) <= 10)[0]
                                                   )
        )
    return bean_num


def _adjacentDist(pacmanPos, ghostPos, type, adjacent_data, locs_df):
    # print("pre adjacent", pacmanPos)

    if isinstance(ghostPos, list):
        ghostPos = ghostPos[0]

    if pacmanPos == (0, 18) or pacmanPos == (-1, 18) or pacmanPos == (-2, 18):
        pacmanPos = (1, 18)
    if pacmanPos == (31, 18) or pacmanPos == (30, 18):
        pacmanPos = (29, 18)
    if isinstance(adjacent_data[pacmanPos][type], float):
        return inf_val

    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent positions in the tunnel
    if adjacent == (0, 18) or adjacent == (-1, 18) or adjacent == (-2, 18):
        adjacent = (1, 18)
    if adjacent == (31, 18) or adjacent == (30, 18):
        adjacent = (29, 18)
    return 0 if adjacent == ghostPos else locs_df[adjacent][ghostPos]


def _ghostModeDist(ifscared1, ifscared2, PG1, PG2, mode):
    if mode == "normal":
        ifscared1 = ifscared1.apply(lambda x: x < 3)
        ifscared2 = ifscared2.apply(lambda x: x < 3)
    elif mode == "scared":
        ifscared1 = ifscared1.apply(lambda x: x > 3)
        ifscared2 = ifscared2.apply(lambda x: x > 3)
    else:
        raise ValueError("Undefined ghost mode {}!".format(mode))
    res = []
    for i in range(ifscared1.shape[0]):
        ind = np.where(np.array([ifscared1[i], ifscared2[i]]) == True)[0]
        res.append(np.min(np.array([PG1[i], PG2[i]])[ind]) if len(ind) > 0 else inf_val)
    return pd.Series(res)


def area(Pos, an=0):
    Pos = list(Pos)
    result = 0
    for pos in Pos:
        if not isinstance(pos, tuple):
            continue
        x, y = pos
        if an == 1 and ((x >= 2 and x <= 7 and y >= 5 and y <= 17) or (x >= 8 and x <= 14 and y >= 5 and y <= 9)):
            result += 1
        if an == 2 and ((x >= 15 and x <= 21 and y >= 5 and y <= 9) or (x >= 22 and x <= 27 and y >= 5 and y <= 17)):
            result += 1
        if an == 3 and (x >= 8 and x <= 21 and y >= 10 and y <= 23):
            result += 1
        if an == 4 and ((x >= 2 and x <= 7 and y >= 19 and y <= 33) or (x >= 8 and x <= 14 and y >= 24 and y <= 33)):
            result += 1
        if an == 5 and ((x >= 15 and x <= 21 and y >= 24 and y <= 33) or (x >= 22 and x <= 27 and y >= 19 and y <= 33)):
            result += 1
        if an == 0 and (y == 18 and ((x >= 0 and x <= 7) or (x >= 22 and x <= 29))):
            result += 1
    return result


def extractFeature(trial, date):
    for i in range(6):
        trial["GA" + str(i)] = trial[["ghost1Pos", "ghost2Pos"]].apply(
            lambda x: area(Pos=x, an=i),
            axis=1)
        trial["PA" + str(i)] = trial[["pacmanPos"]].apply(
            lambda x: area(Pos=x, an=i),
            axis=1)
    if date == "Year1":
        locs_df = _readLocDistance("../ConstantData/dij_distance_map_before.csv")
        adjacent_data = _readAdjacentMap("../ConstantData/adjacent_map_before.csv")
    else:
        locs_df = _readLocDistance("../ConstantData/dij_distance_map.csv")
        adjacent_data = _readAdjacentMap("../ConstantData/adjacent_map.csv")
    trial = trial.reset_index(drop=True)

    eat_energizer = []
    for i in range(len(trial) - 1):
        if isinstance(trial["energizers"][i + 1], float) and isinstance(trial["energizers"][i], list):
            eat_energizer.append(True)
        elif isinstance(trial["energizers"][i + 1], list) and isinstance(trial["energizers"][i], list) and len(
                trial["energizers"][i]) > len(trial["energizers"][i + 1]):
            eat_energizer.append(True)
        else:
            eat_energizer.append(False)
    eat_energizer.append(False)

    # fill empty list
    for colName, content in trial.iteritems():
        trial[colName] = trial[colName].apply(lambda x: float(0) if x == [] else x)

    def distance(keys):
        if keys[1] == "energizers":
            P_left = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else np.min(
                    [_adjacentDist(x[keys[0]], each, "left", adjacent_data, locs_df) for each in x[keys[1]]]),
                axis=1
            )
            P_right = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else np.min(
                    [_adjacentDist(x[keys[0]], each, "right", adjacent_data, locs_df) for each in x[keys[1]]]),
                axis=1
            )
            P_up = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else np.min(
                    [_adjacentDist(x[keys[0]], each, "up", adjacent_data, locs_df) for each in x[keys[1]]]),
                axis=1
            )
            P_down = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else np.min(
                    [_adjacentDist(x[keys[0]], each, "down", adjacent_data, locs_df) for each in x[keys[1]]]),
                axis=1)
        else:
            P_left = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else _adjacentDist(x[keys[0]], x[keys[1]], "left", adjacent_data, locs_df),
                axis=1
            )
            P_right = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else _adjacentDist(x[keys[0]], x[keys[1]], "right", adjacent_data, locs_df),
                axis=1
            )
            P_up = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else _adjacentDist(x[keys[0]], x[keys[1]], "up", adjacent_data, locs_df),
                axis=1
            )
            P_down = trial[keys].apply(
                lambda x: inf_val if isinstance(x[keys[1]], float)
                else _adjacentDist(x[keys[0]], x[keys[1]], "down", adjacent_data, locs_df),
                axis=1)

        P = np.array([P_left, P_right, P_up, P_down])
        P = np.min(P, axis=0)
        return P

    # Features for the estimation
    PG1 = distance(["pacmanPos", "ghost1Pos"])
    PG2 = distance(["pacmanPos", "ghost2Pos"])
    PE = distance(["pacmanPos", "energizers"])

    beans_10step = trial[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float)
        else len(
            np.where(
                np.array([0 if x.pacmanPos == each
                          else locs_df[x.pacmanPos][each] for each in x.beans]) < 10
            )[0]
        ),
        axis=1
    )

    beans_over_10step = trial[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float)
        else len(
            np.where(
                np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) > 10
            )[0]
        ),
        axis=1
    )

    ifscared1 = trial.ifscared1
    ifscared2 = trial.ifscared2

    processed_trial_data = pd.DataFrame(
        data=
        {
            "file": trial.DayTrial,
            "ifscared1": ifscared1,
            "ifscared2": ifscared2,

            "PG1": PG1,
            "PG2": PG2,

            "PE": PE,

            "beans_within_5": beans_10step,
            "beans_beyond_10": beans_over_10step,

            "PA0": trial["PA0"],
            "PA1": trial["PA1"],
            "PA2": trial["PA2"],
            "PA3": trial["PA3"],
            "PA4": trial["PA4"],
            "PA5": trial["PA5"],

            "GA0": trial["GA0"],
            "GA1": trial["GA1"],
            "GA2": trial["GA2"],
            "GA3": trial["GA3"],
            "GA4": trial["GA4"],
            "GA5": trial["GA5"],

            "EE": eat_energizer,
            "weight": trial["weight"],
            "true_dir": trial.pacman_dir,
        }
    )
    return processed_trial_data


def combine_evade(data1, data):
    # evade = []
    PG = []
    IS = []
    for i in range(len(data)):
        index1 = np.array(data[["PG1", "PG2"]].iloc[i])
        index1 = np.argmin(index1)
        if index1 == 0:
            PG.append(data1["PG1"].iloc[i])
            IS.append(data1["IS1"].iloc[i])
        elif index1 == 1:
            PG.append(data1["PG2"].iloc[i])
            IS.append(data1["IS2"].iloc[i])

    data1["PG"] = np.array(PG, dtype=np.int)
    data1["IS"] = np.array(IS, dtype=np.int)
    return data1


def predictor4Prediction(feature_data):
    dir_list = ["left", "right", "up", "down"]
    df = feature_data.copy()

    df.loc[df.ifscared1 == 3, "PG1"] = 100
    df.loc[df.ifscared2 == 3, "PG2"] = 100

    df["if_exist1"] = (df.ifscared1 != -1).astype(int)
    df["if_normal1"] = (df.ifscared1 <= 2).astype(int)
    df["if_dead1"] = (df.ifscared1 == 3).astype(int)
    df["if_scared1"] = (df.ifscared1 >= 4).astype(int)

    df["if_exist2"] = (df.ifscared2 != -1).astype(int)
    df["if_normal2"] = (df.ifscared2 <= 2).astype(int)
    df["if_dead2"] = (df.ifscared2 == 3).astype(int)
    df["if_scared2"] = (df.ifscared2 >= 4).astype(int)

    is_encode = pd.DataFrame()
    for ghost in [1, 2]:
        cols = ["if_" + i + str(ghost) for i in ["normal", "dead", "scared"]]
        is_encode["IS" + str(ghost)] = np.argmax(df[cols].values, 1)

    numerical_cols1 = ["PG1", "PG2", "PE"]
    bin = [0, 11, 101]
    numerical_encode1 = pd.concat(
        [pd.cut(df[i], bin, right=False, labels=[0, 1]) for i in numerical_cols1
         ],
        axis=1,
    )
    numerical_encode1.columns = numerical_cols1

    numerical_encode2 = pd.DataFrame()
    df["beans_within_5"] = np.array(df["beans_within_5"]) / np.max(df["beans_within_5"])
    df["beans_beyond_10"] = np.array(df["beans_beyond_10"]) / np.max(df["beans_beyond_10"])
    numerical_encode2["BN5"] = 1 - np.array(df["beans_within_5"] == 0, dtype=np.int)
    numerical_encode2["BN10"] = 1 - np.array(df["beans_beyond_10"] == 0, dtype=np.int)

    predictors = pd.concat([numerical_encode1, numerical_encode2, is_encode], axis=1)
    predictors = combine_evade(predictors, df)

    A_keys = ["PA0", "PA1", "PA2", "PA3", "PA4", "PA5", "GA0", "GA1", "GA2", "GA3", "GA4", "GA5"]
    predictors[A_keys] = feature_data[A_keys]
    keys = ['PG', 'PG1', 'PG2', 'PE', 'BN5', 'BN10', 'IS', 'IS1', 'IS2'] + A_keys
    predictors = predictors[keys]
    for i in predictors.columns:
        predictors[i] = predictors[i].astype(int)
    return predictors


def main(path, date):
    filename = path.split("/")[-1]
    data = pd.read_pickle(path)
    for colName, content in data.iteritems():
        data[colName] = data[colName].apply(lambda x: float(0) if x == [] else x)

    print("Data shape : ", data.shape)
    # Extract attribute data
    features = extractFeature(data, date)
    features['revise_weight'] = np.array(data['revise_weight'])
    features['contribution'] = np.array(data['contribution'])
    features['weight'] = np.array(data['weight'])
    features['file'] = np.array(data['file'])
    features['level_0'] = np.array(data['level_0'])
    features['strategy'] = np.array(data['strategy'])

    features.reset_index(drop=True, inplace=True)
    # predictors for zzh
    predictors = predictor4Prediction(features)
    predictors['EE'] = np.array(features['EE'])
    predictors['revise_weight'] = np.array(data['revise_weight'])
    predictors['contribution'] = np.array(data['contribution'])
    predictors['weight'] = np.array(data['weight'])
    predictors['file'] = np.array(data['file'])
    predictors['level_0'] = np.array(data['level_0'])
    predictors['strategy'] = np.array(data['strategy'])

    savepath = "../ConstantData/DiscreteFeatureData/" + date + "/" + filename
    predictors.to_pickle(savepath)
    savepath = "../ConstantData/FeatureData/" + date + "/" + filename
    features.to_pickle(savepath)

    strategy = list(predictors["strategy"])
    strategy = [s for i, s in enumerate(strategy) if i == 0 or s != strategy[i - 1]]
    index = np.where(np.array(strategy) == 7)[0]

    print(path, len(index) / len(strategy))


def extract_monkey_feature(date):
    filefolder = "../MonkeyData/CorrectedWeightData/" + date + "/"
    filenames = os.listdir(filefolder)
    filenames = [filefolder + filename for filename in filenames]

    with multiprocessing.Pool(processes=12) as pool:
        pool.map(
            partial(main, date=date), filenames)


if __name__ == '__main__':
    date = "Year3"
    extract_monkey_feature(date)
