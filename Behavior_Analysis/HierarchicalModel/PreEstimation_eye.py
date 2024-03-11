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
import warnings

warnings.filterwarnings("ignore")
import sys
import os

current_directory = os.path.dirname(os.path.abspath("./UtilityConvert.py"))
root_path = os.path.abspath(os.path.dirname(os.path.dirname(current_directory)) + os.path.sep + ".")
sys.path.append(root_path)

from Utils.FileUtils_fmri import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from Behavior_Analysis.HierarchicalModel.Agent.LocalAgent import PathTree as LocalAgent
from Behavior_Analysis.HierarchicalModel.Agent.EvadeAgent_fmri import EvadeTree as EvadeAgent
from Behavior_Analysis.HierarchicalModel.Agent.GlobalAgent_beyond10 import SimpleGlobal as GlobalAgent
from Behavior_Analysis.HierarchicalModel.Agent.ApproachAgent import ApproachTree as ApproachAgent
from Behavior_Analysis.HierarchicalModel.Agent.EnergizerAgent import EnergizerTree as EnergizerAgent
from Behavior_Analysis.HierarchicalModel.Agent.NoEnergizerAgent import NoEnerTree as NoEnerAgent

# ==================================================
# define if the data is fitted with 7 strategies (or 6)
is_7str = 1
# define if the data is derived from human (or monkey)
is_human = 0


# ==================================================
def tran(x):
    if isinstance(x, list):
        x = x[0]
    return x


def _readData(filename):
    '''
    Read data for pre-estimation.
    '''
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
        # all_data = pd.concat([each[1] for each in all_data]).reset_index(drop=True)
    # print(file)
    all_data = all_data.reset_index(drop=True)
    # if isinstance(all_data['fruitPos'].iloc[0],list):
    #     all_data['fruitPos']=all_data['fruitPos'].apply(tran)

    return all_data


def _readAuxiliaryData():
    '''
    Read auxiliary data for the pre-estimation.
    :return:
    '''
    # Load data
    locs_df = readLocDistance("../../Data/constant/dij_distance_map_fmri.csv")
    adjacent_data = readAdjacentMap("../../Data/constant/adjacent_map_fmri.csv")
    adjacent_path = readAdjacentPath("../../Data/constant/dij_distance_map_fmri.csv")
    reward_amount = readRewardAmount()
    return adjacent_data, locs_df, adjacent_path, reward_amount


def _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount, filename):
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
    # Configuration (the last direction)
    if is_7str == 1:
        # Configuration (for no-energizer agent)
        no_energizer_depth = 8
        noener_ghost_attractive_thr = 10
        noener_fruit_attractive_thr = 10
        noener_ghost_repulsive_thr = 10
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Q-value (utility)
    global_Q = []
    local_Q = []
    evade_blinky_Q = []
    evade_clyde_Q = []
    evade_ghost3_Q = []
    evade_ghost4_Q = []
    approach_Q = []
    energizer_Q = []
    if is_7str == 1:
        no_energizer_Q = []
    num_samples = all_data.shape[0]
    print("Sample Num : ", num_samples)
    for index in range(num_samples):
        # print(index)
        if 0 == (index + 1) % 500:
            print("Finished estimation at {}".format(index + 1))
        # Extract game status and PacMan status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        # The tunnel--fMRI version change the columns' number
        if cur_pos == (0, 18) or cur_pos == (-1, 18):
            cur_pos = (1, 18)
        if cur_pos == (30, 18) or cur_pos == (31, 18):
            cur_pos = (29, 18)
        laziness_coeff = 0.0
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([each.ghost1Pos, each.ghost2Pos, each.ghost3Pos, each.ghost4Pos]) \
            if isinstance(each.ghost1Pos, tuple) \
            else np.array([eval(each.ghost1Pos), eval(each.ghost2Pos), eval(each.ghost3Pos), eval(each.ghost4Pos)])
        # ghosts born in wall -- fMRI version change the location
        for gh in range(4):
            if tuple(ghost_data[gh]) == (14, 20):
                ghost_data[gh] = (14, 19)
            if tuple(ghost_data[gh]) == (15, 20):
                ghost_data[gh] = (15, 19)
            if tuple(ghost_data[gh]) == (16, 20):
                ghost_data[gh] = (16, 19)

        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared2, each.ifscared3, each.ifscared4])
        if "fruitType" in all_data.columns.values:
            reward_type = int(each.fruitType) if not np.isnan(each.fruitType) else np.nan
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
        # Evade(ghost3) agent
        evade_ghost3_agent = EvadeAgent(
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
            "ghost3",
            depth=pessimistic_depth,
            ghost_attractive_thr=0,
            fruit_attractive_thr=0,
            ghost_repulsive_thr=0,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=0.0,
            risk_coeff=1.0
        )
        evade_ghost3_result = evade_ghost3_agent.nextDir(return_Q=True)
        evade_ghost3_Q.append(copy.deepcopy(evade_ghost3_result[1]))
        # Evade(ghost4) agent
        evade_ghost4_agent = EvadeAgent(
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
            "ghost4",
            depth=pessimistic_depth,
            ghost_attractive_thr=0,
            fruit_attractive_thr=0,
            ghost_repulsive_thr=0,
            randomness_coeff=randomness_coeff,
            laziness_coeff=laziness_coeff,
            reward_coeff=0.0,
            risk_coeff=1.0
        )
        evade_ghost4_result = evade_ghost4_agent.nextDir(return_Q=True)
        evade_ghost4_Q.append(copy.deepcopy(evade_ghost4_result[1]))
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
    all_data["evade_ghost3_Q"] = np.tile(np.nan, num_samples)
    all_data["evade_ghost3_Q"] = all_data["evade_ghost3_Q"].apply(np.array)
    all_data["evade_ghost3_Q"] = evade_ghost3_Q
    all_data["evade_ghost4_Q"] = np.tile(np.nan, num_samples)
    all_data["evade_ghost4_Q"] = all_data["evade_ghost4_Q"].apply(np.array)
    all_data["evade_ghost4_Q"] = evade_ghost4_Q
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
        print(all_data[["global_Q", "local_Q", "evade_blinky_Q", "evade_clyde_Q", "evade_ghost3_Q", "evade_ghost4_Q",
                        "approach_Q", "energizer_Q"]].iloc[:5])
    return all_data


# ==================================================


def preEstimation_parallelize(filename, adjacent_data, locs_df, adjacent_path, reward_amount, save_base):
    print("-" * 50)
    print(filename)
    all_data = _readData(filename)
    print("Finished reading data.")
    print("Start estimating...")
    all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
    with open("{}/{}-with_Q.pkl".format(save_base, filename.split("/")[-1].split(".")[0]), "wb") as file:
        pickle.dump(all_data, file)
    print("{}-with_Q.pkl saved!".format(filename.split("/")[-1].split(".")[0]))


def preEstimation(filename_list, save_base):
    pd.options.mode.chained_assignment = None
    # Individual Estimation
    print("=" * 15, " Individual Estimation ", "=" * 15)
    adjacent_data, locs_df, adjacent_path, reward_amount = _readAuxiliaryData()
    print("Finished reading auxiliary data.")
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount, filename)
        with open("{}/{}-with_Q.pkl".format(save_base, filename.split("/")[-1].split(".")[0]), "wb") as file:
            pickle.dump(all_data, file)
        print("{}-with_Q.pkl saved!".format(filename.split("/")[-1].split(".")[0]))
    pd.options.mode.chained_assignment = "warn"


def CalculateUtility(filename, saveFolder):
    filename_list = [
        filename
    ]
    preEstimation(filename_list, saveFolder)


if __name__ == '__main__':
    print("eye" * 20)

    fileFolder = "../../eye_data_process/eyeCorrectedTileData/"
    saveFolder = "../../eye_data_process/eyeUtilityData/"
    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + f for f in fileNames]

    exists = []
    tempPaths = [saveFolder + f[:-4] + "-with_Q.pkl" for f in fileNames]
    for filePath in tempPaths:
        exists.append(os.path.exists(filePath))
    index = np.where(np.array(exists) == False)[0]
    filePaths = list(np.array(filePaths)[index])
    print(len(filePaths))
    
    # CalculateUtility(filePaths[0],saveFolder)
    with multiprocessing.Pool(processes=32) as pool:
        result = pool.map(
            partial(CalculateUtility, saveFolder=saveFolder), filePaths)
