'''
Description:
    Model fitting with pre-processed data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
'''
print("=" * 10)
import sys
import torch

sys.path.append("../../")
import numpy as np
import pandas as pd
import pickle
import scipy
import copy
import ruptures as rpt
import os
from sklearn.model_selection import KFold
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import warnings
from Utils.FileUtils import readAdjacentMap, readLocDistance

warnings.filterwarnings("ignore")
from sko.GA import GA

# =================================================
# Global variables
agents = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer"
]  # list of all the agents
all_dir_list = ["left", "right", "up", "down"]
adjacent_data = readAdjacentMap("../ConstantData/adjacent_map_before.csv")
inf_val = 100  # A large number representing the positive infinity
reborn_pos = (14, 27)  # Pacman reborn position

stay_length = 6


def _convertPos(Pos):
    if (Pos == (-1, 18)) or (Pos == (0, 18)):
        Pos = (1, 18)
    if (Pos == (30, 18)) or (Pos == (29, 18)):
        Pos = (28, 18)
    return Pos


def is_available(pos, dir):
    adjacent_pos = adjacent_data[_convertPos(pos)]
    if isinstance(dir, float) or adjacent_pos[dir] is None or isinstance(adjacent_pos[dir], float):
        return False
    else:
        return True


# =================================================

def _makeChoice(prob):
    '''
    Chose a direction based on estimated Q values.
    :param prob: (list) Q values of four directions (lef, right, up, down).
    :return: (int) The chosen direction.
    '''
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = (
                copy_estimated[available_dir_index]
                - np.min(copy_estimated[available_dir_index])
                + 1
        )
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


def _oneHot(val):
    """
    Convert the direction into a one-hot vector.
    :param val: (str) The direction ("left", "right", "up", "down").
    :return: (list) One-hotted vector.
    """
    dir_list = ["left", "right", "up", "down"]
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def _normalize(x):
    '''
    Normalization.
    :param x: (numpy.ndarray) Original data.
    :return: (numpy.ndarray) Normalized data.
    '''
    return (x) / (x).sum()


def _combine(cutoff_pts, dir, eat_energizers):
    '''
    Combine cut off points when necessary.
    '''
    if len(cutoff_pts) > 1:
        temp_pts = [cutoff_pts[0]]
        for i in range(1, len(cutoff_pts)):
            if cutoff_pts[i][1] - cutoff_pts[i][0] > 3:
                if np.all(dir.iloc[cutoff_pts[i][0]:cutoff_pts[i][1]].apply(lambda x: isinstance(x, float)) == True):
                    temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
                else:
                    temp_pts.append(cutoff_pts[i])
            else:
                temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
        cutoff_pts = temp_pts
    return cutoff_pts


def _positivePessi(pess_Q, offset, pos):
    '''
    Make evade agent Q values non-negative.
    '''
    non_zero = []
    if pos == (30, 18) or pos == (31, 18) or pos == (32, 18):
        pos = (30, 18)
    if pos == (0, 18) or pos == (-1, 18) or pos == (-2, 18):
        pos = (1, 18)
    for dir in all_dir_list:
        if None != adjacent_data[pos][dir] and not isinstance(adjacent_data[pos][dir], float):
            non_zero.append(all_dir_list.index(dir))
    pess_Q[non_zero] = pess_Q[non_zero] - offset
    return _normalizeWithInf(pess_Q)


# =================================================

def negativeLikelihood(
        param, all_data, true_prob, agents_list, return_trajectory=False, suffix="_Q"
):
    """
    Compute the negative log-likelihood.
    :param param: (list) Model parameters, which are agent weights.
    :param all_data: (pandas.DataFrame) A table of data.
    :param agent_list: (list) Names of all the agents.
    :param return_trajectory: (bool) Set to True when making predictions.
    :return: (float) Negative log-likelihood
    """
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute negative log likelihood
    # agent_weight = [0, 1, 0, 0, 0, 1]
    nll = 0
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    # raise KeyboardInterrupt
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    dir_Q_value = agent_Q_value @ agent_weight
    dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    exp_prob = np.exp(dir_Q_value)
    accuracy = 0
    for each_sample in range(num_samples):
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        # log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        # nll = nll - log_likelihood
        temp = dir_Q_value[each_sample]
        td = true_dir[each_sample]
        if np.isinf(temp[td]):
            accuracy += 0
        else:
            MAX = np.max(temp)
            MAXS = np.where(temp == MAX)[0]
            if temp[td] == MAX:
                accuracy += 1 / len(MAXS)
            else:
                accuracy += 0
    nll = (1 - accuracy / num_samples) * 1000 + np.sum(np.abs(agent_weight))
    # print(nll)
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def negativeLikelihoodMergeGroup(
        param, all_data, true_prob, group_idx, agents_list, return_trajectory=False, suffix="_Q"
):
    """
    Compute the negative log-likelihood.
    :param param: (list) Model parameters, which are agent weights.
    :param all_data: (pandas.DataFrame) A table of data.
    :param agent_list: (list) Names of all the agents.
    :param return_trajectory: (bool) Set to True when making predictions.
    :return: (float) Negative log-likelihood
    """
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute negative log likelihood
    nll = 0
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    # raise KeyboardInterrupt
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    # merge Q values
    merg_Q_values = np.zeros((agent_Q_value.shape[0], agent_Q_value.shape[1], len(group_idx)))
    for i, g in enumerate(group_idx):
        merg_Q_values[:, :, i] = np.nanmean(agent_Q_value[:, :, g], axis=-1)
    dir_Q_value = merg_Q_values @ agent_weight
    dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values

    # dirs=[]
    # for i, g in enumerate(group_idx):
    #     dir = np.sum(np.argmax(merg_Q_values[:, :, i], axis=1) == true_dir)
    #     dirs.append(dir)
    exp_prob = np.exp(dir_Q_value)
    # for i in range(len(exp_prob)):
    #     exp_prob[i]/=np.sum(exp_prob,axis=1)[i]
    for each_sample in range(num_samples):
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        nll = nll - log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def caculate_correct_rate(result_x, all_data, true_prob, agents, suffix="_Q"):
    '''
    Compute the estimation correct rate of a fitted model.
    '''
    _, estimated_prob = negativeLikelihood(
        result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    correct_rate_ = 0
    for i in range(1000):
        estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir) / len(estimated_dir)
        correct_rate_ += correct_rate
    correct_rate_ = correct_rate_ / 1000
    return correct_rate_


def _calculate_is_correct(result_x, all_data, true_prob, agents, suffix="_Q"):
    '''
    Determine whether the estimation of each time step is correct.
    '''
    _, estimated_prob = negativeLikelihood(
        result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
    is_correct = (estimated_dir == true_dir)
    return is_correct, estimated_dir


def change_dir_index(x):
    '''
    Find the position where the Pacman changes its direction.
    '''
    temp = pd.Series((x != x.shift()).where(lambda x: x == True).dropna().index)
    temp = temp[(temp - temp.shift()) > 1].values
    if len(temp) > 0 and temp[-1] != len(x):
        temp = np.array(list(temp) + [len(x)])
    if len(temp) == 0:
        temp = [len(x)]
    return temp


def fitting_weight_ga_parallelize(idx, cutoff_pts, is_nan, df_monkey, suffix, agents):
    prev = cutoff_pts[idx][0]
    end = cutoff_pts[idx][1]
    print(prev, end)
    if is_nan[idx] == True:
        result = {
            "resultlist": [0] * len(agents) + [0] + [prev] + [end],
            "ind": None,
            "phase_is_correct": None,
            "predict_dir": None,
            "is_vague": False,
            "loss": None
        }
        return result
    all_data = df_monkey[prev:end]
    temp_data = copy.deepcopy(all_data)
    temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
    all_data = all_data[temp_data.nan_dir == False]
    if all_data.shape[0] == 0:
        print("All the directions are nan from {} to {}!".format(prev, end))
        return None
    ind = np.where(temp_data.nan_dir == False)[0] + prev
    true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]

    def Likelihood(agent_weight):
        dir_Q_value = agent_Q_value @ agent_weight
        dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
        accuracy = 0
        for each_sample in range(num_samples):
            if np.isnan(dir_Q_value[each_sample][0]):
                continue
            temp = dir_Q_value[each_sample]
            td = true_dir[each_sample]
            if np.isinf(temp[td]):
                accuracy += 0
            else:
                MAX = np.max(temp)
                MAXS = np.where(temp == MAX)[0]
                if temp[td] == MAX:
                    accuracy += 1 / len(MAXS)
                else:
                    accuracy += 0
        nll = - accuracy / num_samples + 0.1 * np.sum(np.abs(agent_weight))
        return nll

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ga = GA(func=Likelihood, n_dim=len(agents), size_pop=100, max_iter=500, prob_mut=0.01, lb=[0] * len(agents),
            ub=[1] * len(agents),
            precision=1e-3)
    # ga.to(device=device)
    weight, loss = ga.run(500)
    cr = caculate_correct_rate(weight, all_data, true_prob, agents, suffix=suffix)
    index = np.argmax(weight)
    temp_weight = [0] * len(agents)
    temp_weight[index] = 1
    cr_ = caculate_correct_rate(np.array(temp_weight), all_data, true_prob, agents, suffix=suffix)
    vague = False
    if cr_ <= 0.51:
        vague = True

    phase_is_correct, predict_dir = _calculate_is_correct(weight, all_data, true_prob, agents, suffix=suffix)
    result = {
        "resultlist": weight.tolist() + [cr] + [prev] + [end],
        "ind": ind,
        "phase_is_correct": phase_is_correct,
        "predict_dir": predict_dir,
        "is_vague": vague,
        "loss": loss,

    }
    return result


def fit_func(df_monkey, cutoff_pts, is_nan, suffix="_Q", is_match=False,
             agents=agents):
    '''
    Fit model parameters (i.e., agent weights).
    '''
    result_list = []
    total_loss = 0
    is_correct = np.zeros((df_monkey.shape[0],))
    is_correct[is_correct == 0] = np.nan
    pre_dir = np.zeros((df_monkey.shape[0],))
    pre_dir[pre_dir == 0] = np.nan
    is_vague = np.array([False] * len(df_monkey))

    idxs = list(range(len(cutoff_pts)))
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(
            partial(fitting_weight_ga_parallelize, cutoff_pts=cutoff_pts, is_nan=is_nan, df_monkey=df_monkey,
                    suffix=suffix, agents=agents), idxs)
    for i, result in enumerate(results):
        result_list.append(result["resultlist"])
        ind = result["ind"]
        if ind is not None:
            phase_is_correct = result["phase_is_correct"]
            predict_dir = result["predict_dir"]
            is_correct[ind] = phase_is_correct
            pre_dir[ind] = predict_dir
        vague = result["is_vague"]
        if vague is True:
            is_vague[cutoff_pts[i][0]:cutoff_pts[i][1]] = [True] * (cutoff_pts[i][1] - cutoff_pts[i][0])
        loss = result["loss"]
        if loss is not None:
            total_loss += loss
    if is_match:
        return result_list, total_loss, is_correct, pre_dir, is_vague
    else:
        return result_list, total_loss


def merge_fit_func(df_monkey, cutoff_pts, suffix="_Q", is_match=False,
                   agents=["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]):
    '''
    Fit model parameters (i.e., agent weights).
    '''
    agent_Q_list = [each + suffix for each in agents]

    result_list = []
    is_correct = []

    prev = 0
    total_loss = 0
    is_correct = np.zeros((df_monkey.shape[0],))
    is_correct[is_correct == 0] = np.nan
    trial_same_dir_groups = []
    for prev, end in cutoff_pts:
        all_data = df_monkey[prev:end]
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        # -----
        agent_dirs = []
        agent_Q_data = all_data[agent_Q_list]
        for each in agent_Q_list:
            agent_Q_data[each] = agent_Q_data[each].apply(lambda x: np.nan if isinstance(x, list) else x).fillna(
                method="ffill").fillna(
                method="bfill").values
            tmp_dirs = agent_Q_data[each].apply(
                lambda x: np.argmax(x) if not np.all(x[~np.isinf(x)] == 0) else np.nan).values
            agent_dirs.append(tmp_dirs)
        agent_dirs = np.asarray(agent_dirs)
        # -----
        # Split into groups
        wo_nan_idx = np.asarray([i for i in range(agent_dirs.shape[0]) if not np.any(np.isnan(agent_dirs[i]))])
        if len(wo_nan_idx) <= 1:
            same_dir_groups = [[i] for i in range(len(agents))]
        else:
            same_dir_groups = [[i] for i in range(len(agents)) if i not in wo_nan_idx]
            wo_nan_is_same = np.asarray(
                [[np.all(agent_dirs[i] == agent_dirs[j]) for j in wo_nan_idx] for i in wo_nan_idx], dtype=int)
            _, component_labels = connected_components(wo_nan_is_same, directed=False)
            for i in np.unique(component_labels):
                same_dir_groups.append(list(wo_nan_idx[np.where(component_labels == i)[0]]))
        trial_same_dir_groups.append(same_dir_groups)
        # construct reverse table
        reverse_group_idx = {each: [None, 0] for each in range(6)}
        for g_idx, g in enumerate(same_dir_groups):
            for i in g:
                reverse_group_idx[i][0] = g_idx
                for j in g:
                    reverse_group_idx[j][1] += 1
        # -----
        ind = np.where(temp_data.nan_dir == False)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        # -----
        bounds = [[0, 1000] for _ in range(len(same_dir_groups))]
        params = [0.0] * len(same_dir_groups)
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
            u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        func = lambda params: negativeLikelihoodMergeGroup(
            params, all_data, true_prob, same_dir_groups, agents, return_trajectory=False, suffix=suffix
        )
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints=cons,
        )
        if set(res.x) == {0}:
            print("Failed optimization at ({},{})".format(prev, end))
            params = [0.1] * len(same_dir_groups)
            for i, a in enumerate(agents):
                if set(np.concatenate(all_data["{}{}".format(a, suffix)].values)) == {0}:
                    params[i] = 0.0
            res = scipy.optimize.minimize(
                func,
                x0=params,
                method="SLSQP",
                bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
                tol=1e-5,
                constraints=cons,
            )
        total_loss += negativeLikelihoodMergeGroup(
            res.x / res.x.sum(),
            all_data,
            true_prob,
            same_dir_groups,
            agents,
            return_trajectory=False,
            suffix=suffix,
        )
        # -----
        reassigned_weights = [res.x[reverse_group_idx[i][0]] / reverse_group_idx[i][1] for i in range(6)]
        cr = caculate_correct_rate(reassigned_weights, all_data, true_prob, agents, suffix=suffix)
        result_list.append(reassigned_weights + [cr] + [prev] + [end])
        phase_is_correct = _calculate_is_correct(reassigned_weights, all_data, true_prob, agents, suffix=suffix)
        is_correct[ind] = phase_is_correct
    if is_match:
        return result_list, total_loss, is_correct, trial_same_dir_groups
    else:
        return result_list, total_loss


def normalize_weights(result_list, df_monkey):
    '''
    Normalize fitted agent weights.
    '''
    agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
    ]
    df_result = (
        pd.DataFrame(
            result_list,
            columns=[i + "_w" for i in agents] + ["accuracy", "start", "end"],
        )
            .set_index("start")
            .reindex(range(df_monkey.shape[0]))
            .fillna(method="ffill")
    )
    df_plot = df_result.filter(regex="_w").divide(
        df_result.filter(regex="_w").sum(1), 0
    )
    return df_plot, df_result


def merge_context(context, cutoff_pts):
    new_cutoff_pts = [0]
    pointer = 0
    end = -1
    if len(context) > 0:
        for i in range(len(cutoff_pts)):
            if cutoff_pts[i] < context[pointer][0] and cutoff_pts[i] > new_cutoff_pts[-1]:
                new_cutoff_pts.append(cutoff_pts[i])
            elif cutoff_pts[i] >= context[pointer][0] and cutoff_pts[i] <= context[pointer][1]:
                new_cutoff_pts.append(context[pointer][0])
                new_cutoff_pts.append(context[pointer][1])
                pointer += 1
            elif cutoff_pts[i] > context[pointer][1]:
                new_cutoff_pts.append(context[pointer][0])
                new_cutoff_pts.append(context[pointer][1])
                new_cutoff_pts.append(cutoff_pts[i])
                pointer += 1
            end = i
            if pointer >= len(context):
                break
    for i in range(end + 1, len(cutoff_pts)):
        if cutoff_pts[i] > new_cutoff_pts[-1]:
            new_cutoff_pts.append(cutoff_pts[i])
    return new_cutoff_pts[1:]


def add_cutoff_pts(cutoff_pts, df_monkey):
    '''
    Initialize cut-off points at where the ghosts and energizers are eaten.
    '''
    eat_ghost = (
        (
                ((df_monkey.ifscared1 == 3) & (df_monkey.ifscared1.diff() < 0))
                | ((df_monkey.ifscared2 == 3) & (df_monkey.ifscared2.diff() < 0))
        )
            .where(lambda x: x == True)
            .dropna()
            .index.tolist()
    )
    eat_energizers = (
        (
                df_monkey.energizers.apply(
                    lambda x: len(x) if not isinstance(x, float) else 0
                ).diff()
                < 0
        )
            .where(lambda x: x == True)
            .dropna()
            .index.tolist()
    )
    cutoff_pts = sorted(list(cutoff_pts) + eat_ghost + eat_energizers)
    cutoff_pts = list(set(cutoff_pts))
    cutoff_pts.sort()

    # Merge paragraphs that are all nan
    temp = [0 if isinstance(i, float) else 1 for i in df_monkey.next_pacman_dir_fill]
    index = np.where(np.array(temp) == 0)[0]
    if len(index) > 0:
        pre = index[0]
        end = index[0]
        nan_context = []
        count = 1
        for i in range(1, len(index)):
            if index[i] != index[i - 1] + 1:
                if count >= stay_length:
                    end = index[i - 1] + 1
                    nan_context.append((pre, end))
                pre = index[i]
                count = 1
            else:
                count += 1
        cutoff_pts = merge_context(nan_context, cutoff_pts)

    # Merge the ghost-eating paragraphs
    eat_ghost_context = []
    for i, ee in enumerate(eat_energizers):
        pre = ee
        if i == len(eat_energizers) - 1:
            end = len(df_monkey)
        else:
            end = eat_energizers[i + 1]
        last_eat_ghost = None
        for j, eg in enumerate(eat_ghost):
            if eg > pre and eg < end:
                last_eat_ghost = eg
        if last_eat_ghost is not None:
            eat_ghost_context.append((pre, last_eat_ghost))

    cutoff_pts = merge_context(eat_ghost_context, cutoff_pts)
    return cutoff_pts, eat_energizers, eat_ghost

    # return cutoff_pts, eat_energizers


# =================================================

def _normalizeWithInf(x):
    res_x = x.copy()
    tmp_x_idx = np.where(~np.isinf(x))[0]
    if set(x[tmp_x_idx]) == {0}:
        res_x[tmp_x_idx] = 0
    else:
        res_x[tmp_x_idx] = res_x[tmp_x_idx] / np.max(res_x[tmp_x_idx])
    return res_x


def _readData(filename):
    '''
    Read data.
    '''
    print("Filename : ", filename)
    df = pd.read_pickle(filename)
    if "DayTrial" in df.columns.values:
        df["file"] = df.DayTrial
    # -----
    filename_list = df.file.unique()
    selected_data = pd.concat([df[df.file == i] for i in filename_list]).reset_index(drop=True)
    df = selected_data
    # -----------------
    # Drop some columns
    # print(df.columns.values)
    if "global_Q" in df.columns and "global_inf_Q" in df.columns:
        # df = df.drop(columns=['global_Q', 'local_Q', 'evade_blinky_Q', 'evade_clyde_Q', 'approach_Q', 'energizer_Q',
        #                      'global_Q_norm', 'local_Q_norm', 'evade_blinky_Q_norm', 'evade_clyde_Q_norm',
        #                       'approach_Q_norm', 'energizer_Q_norm'])
        df = df.drop(columns=['global_Q', 'local_Q', 'evade_blinky_Q', 'evade_clyde_Q', 'approach_Q', 'energizer_Q',
                              'global_Q_norm', 'local_Q_norm', 'pessimistic_blinky_Q_norm', 'pessimistic_clyde_Q_norm',
                              'suicide_Q_norm', 'planned_hunting_Q_norm'])
        df = df.rename(columns={'global_inf_Q': "global_Q", 'local_inf_Q': 'local_Q',
                                'evade_blinky_inf_Q': 'evade_blinky_Q', 'evade_clyde_inf_Q': 'evade_clyde_Q',
                                'approach_inf_Q': 'approach_Q', 'energizer_inf_Q': 'energizer_Q'})
    # -----------------

    df["game"] = df.file.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )
    data = []
    for idx, grp in df.groupby("game"):
        grp["next_pacman_dir_fill"] = grp["pacman_dir"].shift(-1)
        data.append(copy.deepcopy(grp))
    df = pd.concat(data)
    df.reset_index(inplace=True, drop=True)
    df["next_pacman_dir_fill"] = df.next_pacman_dir_fill.apply(lambda x: x if x is not None else np.nan)

    trial_name_list = np.unique(df.file.values)
    trial_data = []
    for each in trial_name_list:
        pac_dir = df[df.file == each].next_pacman_dir_fill
        if np.sum(pac_dir.apply(lambda x: isinstance(x, float))) == len(pac_dir):
            # all the directions are none
            print("({}) Pacman No Move ! Shape = {}".format(each, pac_dir.shape))
            continue
        else:
            trial_data.append(df[df.file == each])
    df = pd.concat(trial_data).reset_index(drop=True)
    for c in df.filter(regex="_Q").columns:
        if "evade" not in c and ("no_energizer" not in c):
            # df[c + "_norm"] = df[c].apply(
            #     lambda x: x / max(x) if set(x) != {0} else [0, 0, 0, 0]
            # )
            df[c + "_norm"] = df[c].apply(
                lambda x: _normalizeWithInf(x)
            )
        else:
            tmp_val = df[c].explode().values
            offset_num = np.min(tmp_val[tmp_val != -np.inf])
            # offset_num = df[c].explode().min()
            df[c + "_norm"] = df[[c, "pacmanPos"]].apply(
                lambda x: _positivePessi(x[c], offset_num, x.pacmanPos)
                if set(x[c]) != {0}
                else [0, 0, 0, 0],
                axis=1
            )
    return df


def _combinePhases(bkpt_idx, df, return_flag=False):
    agent_list = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]
    agent_Q_list = [each + "_Q_norm" for each in agent_list]
    agent_Q_data = df[agent_Q_list]
    true_dirs = df.next_pacman_dir_fill.fillna(method="bfill").fillna(method="ffill").apply(
        lambda x: ["left", "right", "up", "down"].index(x))
    print("The number of phases (before combination) is {}".format(len(bkpt_idx)))
    # -----
    agent_dirs = []
    for each in agent_Q_list:
        agent_Q_data[each] = agent_Q_data[each].apply(lambda x: np.nan if isinstance(x, list) else x).fillna(
            method="ffill").fillna(
            method="bfill").values
        tmp_dirs = agent_Q_data[each].apply(
            lambda x: np.argmax(x) if not np.all(x[~np.isinf(x)] == 0) else np.nan).values
        agent_dirs.append(tmp_dirs)
    agent_dirs = np.asarray(agent_dirs)
    same_as_true_dir = np.asarray([(agent_dirs[i] == true_dirs).values for i in range(6)])
    phase_same_as_true = [np.nanmean(same_as_true_dir[:, each[0]:each[1]], axis=1) for each in bkpt_idx]
    all_not_same = [np.all(each == 0) for each in phase_same_as_true]
    all_not_same_trial = [[df.file.values[0], bkpt_idx[i]] for i in np.where(all_not_same == True)[0]]
    # -----
    phase_directions = [agent_dirs[:, each[0]:each[1]] for each in bkpt_idx]
    phase_same_num = np.asarray([0 for _ in range(len(bkpt_idx))])
    for p in range(len(bkpt_idx)):
        same_num = 0
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                if i == 2 and j == 3:  # ignore two evade agents
                    continue
                # If there is nan, skip it
                if np.any(np.isnan(phase_directions[p][i, :])) or np.any(np.isnan(phase_directions[p][j, :])):
                    continue
                if np.mean(phase_directions[p][i, :] == phase_directions[p][j, :]) == 1:
                    same_num += 1
        phase_same_num[p] = same_num
    phase_is_same = np.asarray(phase_same_num > 0, dtype=int)
    # -----
    if np.all(phase_is_same == 1):
        new_phase = [(0, bkpt_idx[-1][1])]
    else:
        new_phase = []
        iterator = 0
        while iterator < len(phase_is_same):
            if phase_is_same[iterator] == 0:
                new_phase.append(bkpt_idx[iterator])
            else:
                start_idx = bkpt_idx[iterator][0]
                while phase_is_same[iterator] == 1:
                    iterator += 1
                    if iterator >= len(phase_is_same):  # the last phase has same action sequences
                        start_idx = new_phase[-1][0]
                        end_idx = bkpt_idx[iterator - 1][1]
                        new_phase = new_phase[:-1]
                        break
                    else:
                        end_idx = bkpt_idx[iterator][1]
                new_phase.append((start_idx, end_idx))
            iterator += 1
    print("The number of phases (after combination) is {}".format(len(new_phase)))
    if return_flag:
        return new_phase, phase_is_same
    else:
        return new_phase, all_not_same_trial


def context_event(x, cutoff_pts, eat_energizer, eat_ghost):
    event = []  # 0 means all are nan, 1 means late energizer, 2 means late ghost, 3 means normal
    is_nan = []
    ee_pointer = 0
    eg_pointer = 0
    # and cutoff_pts[i][1] - cutoff_pts[i][0] > 5
    for i in range(len(cutoff_pts)):
        if np.all(x.iloc[cutoff_pts[i][0]:cutoff_pts[i][1]].apply(lambda x: isinstance(x, float)) == True):
            is_nan.append(True)
            event.append(0)
            if ee_pointer < len(eat_energizer) and eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[
                ee_pointer] <= cutoff_pts[i][1]:
                while (eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[ee_pointer] <= cutoff_pts[i][1]):
                    ee_pointer += 1
                    if ee_pointer >= len(eat_energizer):
                        break
            elif ee_pointer < len(eat_energizer) and eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[
                ee_pointer] <= cutoff_pts[i][1]:
                while (eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[ee_pointer] <= cutoff_pts[i][1]):
                    ee_pointer += 1
                    if ee_pointer >= len(eat_energizer):
                        break
        else:
            if ee_pointer < len(eat_energizer) and eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[
                ee_pointer] <= cutoff_pts[i][1]:
                event.append(1)
                while (eat_energizer[ee_pointer] > cutoff_pts[i][0] and eat_energizer[ee_pointer] <= cutoff_pts[i][1]):
                    ee_pointer += 1
                    if ee_pointer >= len(eat_energizer):
                        break
            elif eg_pointer < len(eat_ghost) and eat_ghost[eg_pointer] > cutoff_pts[i][0] and eat_ghost[
                eg_pointer] <= cutoff_pts[i][1]:
                event.append(2)
                while (eat_ghost[eg_pointer] > cutoff_pts[i][0] and eat_ghost[eg_pointer] <= cutoff_pts[i][1]):
                    eg_pointer += 1
                    if eg_pointer >= len(eat_ghost):
                        break
            else:
                event.append(3)
            is_nan.append(False)
    return event, is_nan


def all_directions_nan(data, context):
    pre = context[0]
    end = context[1]
    all_data = data[pre:end]
    temp_data = copy.deepcopy(all_data)
    temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
    all_data = all_data[temp_data.nan_dir == False]
    return all_data.shape[0] == 0


def combine_context(cutoff_pts, event, data):
    nan_index = np.where(np.array(event) == 0)[0]
    need_combine = []
    accept_combine = []

    dict_event_accept = {
        0: [1, 2, 3], 1: [1], 2: [0, 2, 3], 3: [0, 1, 2, 3]
    }  # Each type can accept other types that merge into it

    def is_need_combine(ev, context, data):
        """Determine whether each paragraph needs to be merged"""
        if ev == 0 and context[1] - context[0] >= stay_length:
            return False
        elif ev == 0 and context[1] - context[0] < stay_length:
            return True
        elif ev == 1:
            if all_directions_nan(data, context) == False:
                return False
            else:
                return True
        elif ev == 2:
            if all_directions_nan(data, context) == False:
                return False
            else:
                return True
        elif ev == 3:
            if context[1] - context[0] > 3 and all_directions_nan(data, context) == False:
                return False
            else:
                return True

    for i in range(len(cutoff_pts)):
        need_combine.append(is_need_combine(event[i], cutoff_pts[i], data))
        if event[i] == 0 and (cutoff_pts[i][1] - cutoff_pts[i][0]) >= stay_length:
            accept_combine.append([])
        else:
            accept_combine.append(dict_event_accept[event[i]])

    i = 0
    while i < len(cutoff_pts):
        if need_combine[i] == False:
            pass
        else:
            # If you need to merge, first judge the previous
            status = event[i]
            front_length = np.inf
            tail_length = np.inf
            flag = False
            if i > 0 and status in accept_combine[i - 1]:  # If the previous paragraph can accept the current paragraph
                front_length = cutoff_pts[i - 1][1] - cutoff_pts[i - 1][0]
                flag = True

            if i != len(cutoff_pts) - 1 and status in accept_combine[
                i + 1]:  # If the following paragraph can accept the current paragraph
                tail_length = cutoff_pts[i + 1][1] - cutoff_pts[i + 1][0]
                flag = True
            if flag == False:  # If the current paragraph cannot be accepted before or after
                if all_directions_nan(data, cutoff_pts[
                    i]) == False:  # If the current paragraph can exist independently, it exists independently.
                    need_combine[i] = False
                    i += 1
                    continue
                elif i == len(
                        cutoff_pts) - 1:  # If the current paragraph cannot exist independently, it will be forcibly merged into other paragraphs.
                    front_length = cutoff_pts[i - 1][1] - cutoff_pts[i - 1][0]
                elif i == 0:
                    tail_length = cutoff_pts[i + 1][1] - cutoff_pts[i + 1][0]
                elif event[i - 1] == 1:
                    front_length = cutoff_pts[i - 1][1] - cutoff_pts[i - 1][0]
                elif event[i + 1] == 1:
                    tail_length = cutoff_pts[i + 1][1] - cutoff_pts[i + 1][0]
                else:
                    front_length = cutoff_pts[i - 1][1] - cutoff_pts[i - 1][0]
                    tail_length = cutoff_pts[i + 1][1] - cutoff_pts[i + 1][0]
                    print("front and tail is not!", "=" * 50)
            if front_length < tail_length:  # Select shorter paragraphs to merge
                if event[i - 1] != 0 and event[i] != 0:
                    event[i - 1] = min(event[i - 1], event[i])
                elif event[i - 1] == 0 and event[i] != 0:
                    event[i - 1] = event[i]
                elif event[i - 1] != 0 and event[i] == 0:
                    event[i - 1] = event[i - 1]
                elif event[i - 1] == 0 and event[i] == 0:
                    event[i - 1] = 0
                cutoff_pts[i - 1] = (cutoff_pts[i - 1][0], cutoff_pts[i][1])
                accept_combine[i - 1] = dict_event_accept[event[i - 1]]
                need_combine[i - 1] = is_need_combine(event[i - 1], cutoff_pts[i - 1], data)
                if i + 1 < len(cutoff_pts):
                    cutoff_pts = cutoff_pts[:i] + cutoff_pts[i + 1:]
                    event = event[:i] + event[i + 1:]
                    accept_combine = accept_combine[:i] + accept_combine[i + 1:]
                    need_combine = need_combine[:i] + need_combine[i + 1:]
                else:
                    cutoff_pts = cutoff_pts[:i]
                    event = event[:i]
                    accept_combine = accept_combine[:i]
                    need_combine = need_combine[:i]
                i -= 2
            elif front_length >= tail_length:
                cutoff_pts[i] = (cutoff_pts[i][0], cutoff_pts[i + 1][1])
                # event[i] = min(event[i], event[i + 1])
                if event[i] != 0 and event[i + 1] != 0:
                    event[i] = min(event[i], event[i + 1])
                elif event[i] == 0 and event[i + 1] != 0:
                    event[i] = event[i + 1]
                elif event[i] != 0 and event[i + 1] == 0:
                    event[i] = event[i]
                elif event[i] == 0 and event[i + 1] == 0:
                    event[i] = 0
                accept_combine[i] = dict_event_accept[event[i]]
                need_combine[i] = is_need_combine(event[i], cutoff_pts[i], data)
                if i + 2 < len(cutoff_pts):
                    cutoff_pts = cutoff_pts[:i + 1] + cutoff_pts[i + 2:]
                    event = event[:i + 1] + event[i + 2:]
                    accept_combine = accept_combine[:i + 1] + accept_combine[i + 2:]
                    need_combine = need_combine[:i + 1] + need_combine[i + 2:]
                else:
                    cutoff_pts = cutoff_pts[:i + 1]
                    event = event[:i + 1]
                    accept_combine = accept_combine[:i + 1]
                    need_combine = need_combine[:i + 1]
                i -= 1
        i += 1
    is_nan = []
    for i in event:
        if i == 0:
            is_nan.append(True)
        else:
            is_nan.append(False)

    cannot_fitted = []
    need_combine = []
    for i in range(len(cutoff_pts)):
        need_combine.append(is_need_combine(event[i], cutoff_pts[i], data))
    for i in range(len(cutoff_pts)):
        if need_combine[i] == False:
            cannot_fitted.append(False)
        else:
            if all_directions_nan(data, cutoff_pts[i]) == False:
                cannot_fitted.append(False)
            else:
                cannot_fitted.append(True)
    if np.sum(cannot_fitted) != 0:
        print("need_combine is not 0!", "=" * 50)

    for i in range(len(cutoff_pts) - 1):
        if cutoff_pts[i][1] != cutoff_pts[i + 1][0]:
            print("loss data!", "=" * 50)
    if cutoff_pts[-1][1] != len(data):
        print("loss data!", "=" * 50)
    return cutoff_pts, is_nan


def dynamicStrategyFitting(config):
    '''
    Dynamic strategy model fitting.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    # -----
    print("=== Dynamic Strategy Fitting ====")
    print("Start reading data...")
    df = _readData(config["filename"])
    # df=df.iloc[:2000]
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    all_trial_record = []
    # all_not_same_list = []
    same_group_list = []
    if "index" not in df.columns.values:
        df = df.reset_index(drop=False)
        df = df.reset_index(drop=False)
    elif "level_0" not in df.columns.values:
        df = df.reset_index(drop=False)
    all_context = []
    all_is_nan = []
    all_eat_energizers = []
    all_eat_ghost = []
    for t, trial_name in enumerate(trial_name_list):
        df_monkey = df[df.file == trial_name]
        df_monkey.reset_index(drop=True, inplace=True)
        print("| ({}) {} | Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points

        cutoff_pts, eat_energizers, eat_ghost = add_cutoff_pts(change_dir_index(df_monkey.next_pacman_dir_fill),
                                                               df_monkey)  # Add eating ghost and eating energizer points
        all_eat_energizers += [ea + df_monkey["level_0"].iloc[0] for ea in eat_energizers]
        all_eat_ghost += [eg + df_monkey["level_0"].iloc[0] for eg in eat_ghost]
        cutoff_pts = list(zip([0] + list(cutoff_pts[:-1]), cutoff_pts))

        event, is_nan = context_event(df_monkey.next_pacman_dir_fill, cutoff_pts, eat_energizers, eat_ghost)
        cutoff_pts, is_nan = combine_context(cutoff_pts, event, df_monkey)

        cutoff_pts = [(point[0] + df_monkey["level_0"].iloc[0], point[1] + df_monkey["level_0"].iloc[0]) for point in
                      cutoff_pts]
        all_is_nan += is_nan
        all_context += cutoff_pts
    result_list, total_loss, is_correct, pre_dir, is_vague = fit_func(df, all_context, is_nan=all_is_nan,
                                                                      is_match=True,
                                                                      suffix=suffix,
                                                                      agents=agents)
    is_nan = all_is_nan
    #
    # df = df.reset_index()
    trial_weight = []
    trial_context = []
    trial_contribution = []
    trial_is_stay = []
    for i, res in enumerate(result_list):
        weight = res[:len(agents)]
        prev = res[-2]
        end = res[-1]
        for _ in range(prev, end):
            trial_context.append((prev, end))
            trial_weight.append(weight)
            trial_is_stay.append(is_nan[i])
            if is_nan[i] == False and np.sum(weight) != 0:
                temp_weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
                trial_contribution.append(temp_weight)
            else:
                trial_contribution.append(copy.deepcopy(weight))
    if len(trial_weight) != df.shape[0]:
        # print(config["filename"], "=" * 10, trial_name)
        df["weight"] = [np.nan for _ in range(df.shape[0])]
        df["contribution"] = [np.nan for _ in range(df.shape[0])]
        df["is_correct"] = [np.nan for _ in range(df.shape[0])]
    elif len(trial_weight) > 0:
        df["weight"] = trial_weight
        df["contribution"] = trial_contribution
        df["is_correct"] = is_correct
        df["predict_dir"] = pre_dir
        df["trial_context"] = trial_context
        df["eat_energizer"] = [False] * len(df)
        df["eat_energizer"].iloc[all_eat_energizers] = [True] * len(all_eat_energizers)
        df["eat_ghost"] = [False] * len(df)
        df["eat_ghost"].iloc[all_eat_ghost] = [True] * len(all_eat_ghost)
        df["is_stay"] = trial_is_stay
        df["is_vague"] = is_vague
        print(np.sum(is_vague) / len(df))
    else:
        pass
    # save data
    print("Finished fitting.")
    with open("{}/{}-merge_weight-dynamic-res.pkl".format(config["save_base"],
                                                          config["filename"].split("/")[-1].split(".")[-2]),
              "wb") as file:
        pickle.dump(df, file)
    # np.save("{}/{}-merge_weight-dynamic-bkpt.npy".format(config["save_base"],
    #                                                      config["filename"].split("/")[-1].split(".")[-2]),
    #         best_bkpt_list)
    # np.save("{}/{}-merge_weight-dynamic-same_dir_groups.npy".format(config["save_base"],
    #                                                                 config["filename"].split("/")[-1].split(".")[-2]),
    #         same_group_list)
    print("Finished saving data.")


def fitting_weight_ga(df_monkey, suffix, agents):
    all_data = copy.deepcopy(df_monkey)
    temp_data = copy.deepcopy(all_data)
    temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
    temp_data["available_dir"] = temp_data[['pacmanPos', 'next_pacman_dir_fill']].apply(
        lambda x: is_available(x.pacmanPos, x.next_pacman_dir_fill), axis=1)
    all_data = all_data[(temp_data.nan_dir == False) & (temp_data.available_dir == True)]
    if all_data.shape[0] == 0:
        print("All the directions are nan from {} to {}!")
        return None
    ind = np.where((temp_data.nan_dir == False) & (temp_data.available_dir == True))[0]
    true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]

    def Likelihood(agent_weight):
        dir_Q_value = agent_Q_value @ agent_weight
        dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
        accuracy = 0
        for each_sample in range(num_samples):
            if np.isnan(dir_Q_value[each_sample][0]):
                continue
            temp = dir_Q_value[each_sample]
            td = true_dir[each_sample]
            if np.isinf(temp[td]):
                accuracy += 0
            else:
                MAX = np.max(temp)
                MAXS = np.where(temp == MAX)[0]
                if temp[td] == MAX:
                    accuracy += 1 / len(MAXS)
                else:
                    accuracy += 0
        nll = - accuracy / num_samples + 0.1 * np.sum(np.abs(agent_weight))
        return nll

    from sko.GA import GA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ga = GA(func=Likelihood, n_dim=len(agents), size_pop=100, max_iter=10, prob_mut=0.01, lb=[0] * len(agents),
            ub=[1] * len(agents),
            precision=1e-3)
    # ga.to(device=device)
    weight, loss = ga.run(10)
    cr = caculate_correct_rate(weight, all_data, true_prob, agents, suffix=suffix)

    return weight, cr


def staticStrategyFitting(config):
    '''
    Static strategy model fitting.
    '''
    print("=== Static Strategy Fitting ====")
    filenames = config["filename"]
    df = []
    for filename in filenames:
        d = _readData(filename)
        df.append(copy.deepcopy(d))
    df = pd.concat(df)
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    all_data = copy.deepcopy(df)
    print("Shape of data : ", all_data.shape)

    is_correct = np.zeros((all_data.shape[0],))
    is_correct[is_correct == 0] = np.nan

    temp_data = copy.deepcopy(all_data)
    temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
    all_data = all_data[temp_data.nan_dir == False]
    true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
    ind = np.where(temp_data.nan_dir == False)[0]

    # Model selection with 5-fold cross-validation
    kf = KFold(n_splits=5)
    all_res = {}
    index = 1
    for train_index, test_index in kf.split(all_data):
        X_train, X_test = all_data.iloc[train_index], all_data.iloc[test_index]
        y_train, y_test = true_prob.iloc[train_index], true_prob.iloc[test_index]
        X_train.reset_index(drop=True, inplace=True)
        weight, cr = fitting_weight_ga(df_monkey=X_train, suffix=suffix,
                                       agents=agents)
        cr = caculate_correct_rate(weight, X_test, y_test, agents, suffix=suffix)
        all_res[index] = [cr, copy.deepcopy(weight)]
        print("|Fold {}| Avg correct rate : ".format(index), cr, weight)
        index += 1
    print("Finished fitting.")
    print("=" * 50)
    best_model = sorted([[k, all_res[k][0]] for k in all_res], key=lambda x: x[1])
    print("Best model index is ", best_model[-1][0])
    best_par = all_res[best_model[-1][0]][1]
    phase_is_correct, predict_dir = _calculate_is_correct(best_par, all_data, true_prob, agents, suffix=suffix)
    is_correct[ind] = phase_is_correct
    df["is_correct"] = is_correct
    df["best_par"] = [best_par] * len(is_correct)

    # np.save("{}/{}-merge_weight-static_is_correct.npy".format(config["save_base"],
    #                                                           config["filename"].split("/")[-1].split(".")[-2]),
    #         is_correct)
    # np.save("{}/{}-merge_weight-static_weight.npy".format(
    #     config["save_base"], config["filename"].split("/")[-1].split(".")[-2]),
    #     best_par)
    with open("{}/{}-merge_weight-static.pkl".format(config["save_base"], config["savename"]),
              "wb") as file:
        pickle.dump(df, file)
    print("Finished saving data.")


from functools import partial
import multiprocessing


def p(filename, saveFolder, savename):
    config = {
        "filename": filename,
        "save_base": saveFolder,
        "savename": savename
    }
    dynamicStrategyFitting(config)


def get_monkey_weight(date):
    global adjacent_data
    if date == "Year1":
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map_before.csv")
    else:
        adjacent_data = readAdjacentMap("../ConstantData/adjacent_map.csv")
    fileFolder = "../MonkeyData/CorrectedUtilityData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    filePaths = [fileFolder + fileName for fileName in fileNames]
    saveFolder = "../MonkeyData/WeightData/" + date + "/"

    with multiprocessing.Pool(processes=20) as pool:
        pool.map(
            partial(p, saveFolder=saveFolder, savename=''), filePaths)


if __name__ == '__main__':
    date = "Year3"
    get_monkey_weight(date)
