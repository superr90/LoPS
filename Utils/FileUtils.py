'''
Description:
    Tool functions for the analysis.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np


def readAdjacentMap(filename):
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
    # if (-1, 18) not in dict_adjacent_data:
    #     dict_adjacent_data[(-1, 18)] = {}
    if (0, 18) not in dict_adjacent_data:
        dict_adjacent_data[(0, 18)] = {}
    if (29, 18) not in dict_adjacent_data:
        dict_adjacent_data[(29, 18)] = {}
    # if (30, 18) not in dict_adjacent_data:
    #     dict_adjacent_data[(30, 18)] = {}
    # dict_adjacent_data[(-1, 18)]["left"] = (30, 18)
    # dict_adjacent_data[(-1, 18)]["right"] = (0, 18)
    # dict_adjacent_data[(-1, 18)]["up"] = np.nan
    # dict_adjacent_data[(-1, 18)]["down"] = np.nan
    dict_adjacent_data[(0, 18)]["left"] = (29, 18)
    dict_adjacent_data[(0, 18)]["right"] = (1, 18)
    dict_adjacent_data[(0, 18)]["up"] = np.nan
    dict_adjacent_data[(0, 18)]["down"] = np.nan
    dict_adjacent_data[(29, 18)]["left"] = (28, 18)
    dict_adjacent_data[(29, 18)]["right"] = (0, 18)
    dict_adjacent_data[(29, 18)]["up"] = np.nan
    dict_adjacent_data[(29, 18)]["down"] = np.nan
    # dict_adjacent_data[(30, 18)]["left"] = (29, 18)
    # dict_adjacent_data[(30, 18)]["right"] = (-1, 18)
    # dict_adjacent_data[(30, 18)]["up"] = np.nan
    # dict_adjacent_data[(30, 18)]["down"] = np.nan
    return dict_adjacent_data


def readAdjacentPath(filename):
    adjacent_data = pd.read_csv(filename)
    adjacent_data.pos1 = adjacent_data.pos1.apply(lambda x: eval(x))
    adjacent_data.pos2 = adjacent_data.pos2.apply(lambda x: eval(x))
    adjacent_data.path = adjacent_data.path.apply(lambda x: eval(x))
    return adjacent_data[["pos1", "pos2", "path"]]


def readLocDistance(filename):
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
    dict_locs_df[(0, 18)][(29, 18)] = 1
    # dict_locs_df[(0, 18)][(30, 18)] = 1
    dict_locs_df[(0, 18)][(1, 18)] = 1
    dict_locs_df[(29, 18)][(0, 18)] = 1
    dict_locs_df[(29, 18)][(28, 18)] = 1
    return dict_locs_df


def readRewardAmount():
    '''
    Reward amount for every type of reward
    :return: A dictionary denoting the reward amount of each type of reward.
    '''
    reward_amount = {
        1: 2,  # bean
        2: 4,  # energizer (default as 4)
        3: 3,  # cherry
        4: 5,  # strawberry
        5: 8,  # orange
        6: 12,  # apple
        7: 17,  # melon
        8: 8,  # ghost
        9: 8  # eaten by ghost
    }
    return reward_amount


if __name__ == '__main__':
    locs_df = readLocDistance("../Data/constant/dij_distance_map_before.csv")
    adjacent_data = readAdjacentMap("../Data/constant/adjacent_map_before.csv")
    adjacent_path = readAdjacentPath("../Data/constant/dij_distance_map_before.csv")
    reward_amount = readRewardAmount()
    x = 0
