'''
Description:
    Compute Q values.
    For the efficiency of model fitting, we pre-compute Q-values for data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    17 Dec. 2020
'''

import pickle
import pandas as pd
import numpy as np
import copy
from functools import partial
import multiprocessing
import os
import sys

current_directory = os.path.dirname(os.path.abspath("./UtilityConvert.py"))
root_path = os.path.abspath(os.path.dirname(os.path.dirname(current_directory)) + os.path.sep + ".")
sys.path.append(root_path)
from Utils.FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from Behavior_Analysis.HierarchicalModel.Agent.LocalAgent import PathTree as LocalAgent
from Behavior_Analysis.HierarchicalModel.Agent.EvadeAgent import EvadeTree as EvadeAgent
from Behavior_Analysis.HierarchicalModel.Agent.GlobalAgent_beyond10 import SimpleGlobal as GlobalAgent
from Behavior_Analysis.HierarchicalModel.Agent.ApproachAgent import ApproachTree as ApproachAgent
from Behavior_Analysis.HierarchicalModel.Agent.EnergizerAgent import EnergizerTree as EnergizerAgent
from Behavior_Analysis.HierarchicalModel.Agent.NoEnergizerAgent import NoEnerTree as NoEnerAgent

# ==================================================
# define if the data is fitted with 7 strategies (or 6)
is_7str = 1
# define if the data is derived from human (or monkey)
is_human = 0

global data_time

"""
In this file, the names corresponding to the utility values of different strategies are as follows:
local:local_Q
global:global_Q,
evade:evade_blinkey_Q,evade_clyde_Q
approach:approach_Q,
energizer:energizer_Q,
save:no_energizer_Q,
"""
# specify monkey name if the data is derived from monkey


# ==================================================

def _readData(filename):
    '''
    Read data for pre-estimation.
    '''
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    all_data = all_data.reset_index(drop=True)
    return all_data


def _readAuxiliaryData(date):
    '''
    Read auxiliary data for the pre-estimation.
    :return:
    '''
    # Load data
    if date == "Year1":
        locs_df = readLocDistance("../ConstantData/dij_distance_map_before.csv")
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map_before.csv")
        adjacent_path = readAdjacentPath("../ConstantData/dij_distance_map_before.csv")
    else:
        locs_df = readLocDistance(".../ConstantData/dij_distance_map.csv")
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map.csv")
        adjacent_path = readAdjacentPath("../ConstantData/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    return adjacent_data, locs_df, adjacent_path, reward_amount


def _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount):
    # Randomness
    randomness_coeff = 0.0
    # Configuration (for global agent)
    global_depth = 15
    ignore_depth = 10
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 34
    # Configuration (for local agent)
    local_depth = 10
    local_ghost_attractive_thr = 10
    local_fruit_attractive_thr = 10
    local_ghost_repulsive_thr = 10
    # Configuration (for evade agent)
    pessimistic_depth = 10
    pessimistic_ghost_attractive_thr = 10
    pessimistic_fruit_attractive_thr = 10
    pessimistic_ghost_repulsive_thr = 10
    # Configuration (fpr energizer agent)
    ghost_attractive_thr = 10
    energizer_attractive_thr = 10
    beans_attractive_thr = 10
    # Configuration (for approach agent)
    suicide_depth = 10
    suicide_ghost_attractive_thr = 10
    suicide_fruit_attractive_thr = 10
    suicide_ghost_repulsive_thr = 10
    if is_7str == 1:
        # Configuration (for no-energizer agent)
        no_energizer_depth = 8
        noener_ghost_attractive_thr = 10
        noener_fruit_attractive_thr = 10
        noener_ghost_repulsive_thr = 10
    # Configuration (the last direction)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Q-value (utility)
    global_Q = []
    local_Q = []
    evade_blinky_Q = []
    evade_clyde_Q = []
    approach_Q = []
    energizer_Q = []
    if is_7str == 1:
        no_energizer_Q = []
    num_samples = all_data.shape[0]
    print("Sample Num : ", num_samples)
    for index in range(num_samples):
        # if index >= 2660:
        #     xxx = 0
        # else:
        #     continue
        if 0 == (index + 1) % 20:
            print("Finished estimation at {}".format(index + 1))
        # print(index)
        # Extract game status and PacMan status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        # The tunnel
        if cur_pos == (0, 18) or cur_pos == (-1, 18):
            cur_pos = (1, 18)
        if cur_pos == (29, 18) or cur_pos == (30, 18):
            cur_pos = (28, 18)
        laziness_coeff = 0.0
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        if tuple(ghost_data[0]) == (14, 20):
            ghost_data[0] = (14, 19)
        if tuple(ghost_data[0]) == (15, 20):
            ghost_data[0] = (15, 19)
        if tuple(ghost_data[1]) == (14, 20):
            ghost_data[1] = (14, 19)
        if tuple(ghost_data[1]) == (15, 20):
            ghost_data[1] = (15, 19)
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared2])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
        # Global agents
        global_agent = GlobalAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            depth=global_depth,
            ignore_depth=ignore_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=1.0,
            risk_coeff=0.0
        )
        global_result = global_agent.nextDir(return_Q=True)
        global_Q.append(copy.deepcopy(global_result[1]))
        # Local estimation
        local_agent = LocalAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            depth=local_depth,
            ghost_attractive_thr=local_ghost_attractive_thr,
            fruit_attractive_thr=local_fruit_attractive_thr,
            ghost_repulsive_thr=local_ghost_repulsive_thr,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=1.0,
            risk_coeff=0.0
        )
        local_result = local_agent.nextDir(return_Q=True)
        local_Q.append(copy.deepcopy(local_result[1]))
        # Evade(Blinky) agent
        evade_blinky_agent = EvadeAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            "blinky",
            depth=pessimistic_depth,
            ghost_attractive_thr=0,
            fruit_attractive_thr=0,
            ghost_repulsive_thr=0,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=0.0,
            risk_coeff=1.0
        )
        evade_blinky_result = evade_blinky_agent.nextDir(return_Q=True)
        evade_blinky_Q.append(copy.deepcopy(evade_blinky_result[1]))
        # Evade(Clyde) agent
        evade_clyde_agent = EvadeAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            "clyde",
            depth=pessimistic_depth,
            ghost_attractive_thr=0,
            fruit_attractive_thr=0,
            ghost_repulsive_thr=0,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=0.0,
            risk_coeff=1.0
        )
        evade_clyde_result = evade_clyde_agent.nextDir(return_Q=True)
        evade_clyde_Q.append(copy.deepcopy(evade_clyde_result[1]))
        # Approach agent
        approach_agent = ApproachAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            depth=suicide_depth,
            ghost_attractive_thr=0,
            ghost_repulsive_thr=0,
            fruit_attractive_thr=0,
            randomness_coeff=randomness_coeff,
            # laziness_coeff = laziness_coeff,
            laziness_coeff=0.0,
            reward_coeff=1.0,
            risk_coeff=0.0
        )
        approach_result = approach_agent.nextDir(return_Q=True)
        approach_Q.append(copy.deepcopy(approach_result[1]))
        # Energizer agent
        energizer_agent = EnergizerAgent(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            ghost_attractive_thr=0,
            ghost_repulsive_thr=0,
            fruit_attractive_thr=0,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=1.0,
            risk_coeff=0.0
        )
        energizer_result = energizer_agent.nextDir(return_Q=True)
        energizer_Q.append(copy.deepcopy(energizer_result[1]))
        if is_7str == 1:
            # NoEnergizer agent
            no_energizer_agent = NoEnerAgent(
                adjacent_data,
                locs_df,
                reward_amount,
                cur_pos,
                energizer_data,
                bean_data,
                ghost_data,
                reward_type,
                fruit_pos,
                ghost_status,
                last_dir[index],
                depth=no_energizer_depth,
                ghost_attractive_thr=0,
                fruit_attractive_thr=0,
                ghost_repulsive_thr=0,
                randomness_coeff=randomness_coeff,
                laziness_coeff=laziness_coeff,
                reward_coeff=0.0,
                risk_coeff=1.0
            )
            no_energizer_result = no_energizer_agent.nextDir(return_Q=True)
            no_energizer_Q.append(copy.deepcopy(no_energizer_result[1]))
    # Assign new columns
    print("Estimation length : ", len(global_Q))
    print("Data Shape : ", all_data.shape)
    all_data["global_Q"] = np.tile(np.nan, num_samples)
    all_data["global_Q"] = all_data["global_Q"].apply(np.array)
    all_data["global_Q"] = global_Q
    all_data["local_Q"] = np.tile(np.nan, num_samples)
    all_data["local_Q"] = all_data["local_Q"].apply(np.array)
    all_data["local_Q"] = local_Q
    all_data["evade_blinky_Q"] = np.tile(np.nan, num_samples)
    all_data["evade_blinky_Q"] = all_data["evade_blinky_Q"].apply(np.array)
    all_data["evade_blinky_Q"] = evade_blinky_Q
    all_data["evade_clyde_Q"] = np.tile(np.nan, num_samples)
    all_data["evade_clyde_Q"] = all_data["evade_clyde_Q"].apply(np.array)
    all_data["evade_clyde_Q"] = evade_clyde_Q
    all_data["approach_Q"] = np.tile(np.nan, num_samples)
    all_data["approach_Q"] = all_data["approach_Q"].apply(np.array)
    all_data["approach_Q"] = approach_Q
    all_data["energizer_Q"] = np.tile(np.nan, num_samples)
    all_data["energizer_Q"] = all_data["energizer_Q"].apply(np.array)
    all_data["energizer_Q"] = energizer_Q
    if is_7str == 1:
        all_data["no_energizer_Q"] = np.tile(np.nan, num_samples)
        all_data["no_energizer_Q"] = all_data["no_energizer_Q"].apply(np.array)
        all_data["no_energizer_Q"] = no_energizer_Q
    print("\n")
    print("Direction Estimation :")
    print("\n")
    print("Q value Example:")
    if is_7str == 1:
        print(all_data[["global_Q", "local_Q", "evade_blinky_Q", "evade_clyde_Q", "approach_Q", "energizer_Q",
                        "no_energizer_Q"]].iloc[:5])
    else:
        print(
            all_data[["global_Q", "local_Q", "evade_blinky_Q", "evade_clyde_Q", "approach_Q", "energizer_Q"]].iloc[:5])
    return all_data


# ==================================================

def preEstimation(filename_list, save_base, date):
    pd.options.mode.chained_assignment = None
    # Individual Estimation
    print("=" * 15, " Individual Estimation ", "=" * 15)
    adjacent_data, locs_df, adjacent_path, reward_amount = _readAuxiliaryData(date)
    print("Finished reading auxiliary data.")
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        index1 = np.where(all_data["pacmanPos"] == (23, 30))[0]
        index2 = np.where(all_data["ghost1Pos"] == (23, 30))[0]
        index3 = np.where(all_data["ghost2Pos"] == (23, 30))[0]
        if len(list(index1) + list(index2) + list(index3)) != 0:
            continue
        # print(list(all_data.columns))
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
        with open("{}/{}-with_Q-{}-{}.pkl".format(save_base, filename.split("/")[-1].split(".")[0],
                                                  'human' if is_human == 1 else 'monkey',
                                                  '7str' if is_7str == 1 else '6str'), "wb") as file:
            pickle.dump(all_data, file)
        print("{}/{}-with_Q-{}-{}.pkl saved!".format(save_base, filename.split("/")[-1].split(".")[0],
                                                     'human' if is_human == 1 else 'monkey',
                                                     '7str' if is_7str == 1 else '6str'))
    pd.options.mode.chained_assignment = "warn"


def CalculateUtility(filename, saveFolder, date):
    filename_list = [
        filename
    ]
    preEstimation(filename_list, saveFolder, date)


def get_monkey_utility(date):
    fileFolder = "../MonkeyData/CorrectedTileData/" + date + "/"
    saveFolder = "../MonkeyData/UtilityData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + f for f in fileNames]
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(
            partial(CalculateUtility, saveFolder=saveFolder, date=date), filePaths)


if __name__ == '__main__':
    date = "Year3"
    get_monkey_utility(date)
