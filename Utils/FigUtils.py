'''
Description:
    Utility functions for the figure plotting.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    3 Feb. 2021
'''
import numpy as np
import pandas as pd
import itertools
import random
from more_itertools import consecutive_groups as cg
import copy

import sys
sys.path.append("/home/qlyang/Documents/pacman/")


# =============================================================
#                   CONSTANT VARIABLES
# =============================================================
def eval_df(df_total, l):
    for c in l:
        df_total[c] = df_total[c].apply(
            lambda x: eval(x) if isinstance(x, str) else np.nan
        )
    return df_total

#TODO: remove this part out of this file
try:
	MAP_INFO = eval_df(pd.read_csv("../../Data/constant/map_info_brian.csv"), ["pos", "pos_global"])
except:
	MAP_INFO = eval_df(pd.read_csv("../Data/constant/map_info_brian.csv"), ["pos", "pos_global"])

POSSIBLE_DIRS = (
    MAP_INFO[["pos", "Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"]]
    .replace({0: np.nan})
    .set_index("pos")
)
POSSIBLE_DIRS.columns = ["up", "left", "down", "right"]
POSSIBLE_DIRS = (
    POSSIBLE_DIRS.stack()
    .reset_index(level=1)
    .groupby(level=0, sort=False)["level_1"]
    .apply(list)
    .reset_index()
    .rename(columns={"pos": "p_choice"})
)
GHOST_HOME_POS = [tuple(i) for i in itertools.product(range(12, 18), range(17, 20))] + [
    (14, 16),
    (15, 16),
]
CROSS_POS = MAP_INFO[MAP_INFO.NextNum >= 3].pos.values
CROSS_POS = list(
    set(CROSS_POS)
    - set(
        [
            i
            for i in CROSS_POS
            if i[0] >= 11 and i[0] <= 18 and i[1] >= 16 and i[1] <= 20
        ]
    )
)
TURNING_POS = list(
    set(
        POSSIBLE_DIRS[
            POSSIBLE_DIRS.apply(
                lambda x: sorted(x.level_1) not in [["down", "up"], ["left", "right"]]
                and x.p_choice not in GHOST_HOME_POS,
                1,
            )
        ].p_choice.values.tolist()
        + CROSS_POS
    )
)
OPPOSITE_DIRS = {"left": "right", "right": "left", "up": "down", "down": "up"}

LOCS_DF = eval_df(
    pd.read_csv("../../Data/constant/dij_distance_map.csv"),
    ["pos1", "pos2", "path", "relative_dir"],
)


def relative_dir(pt_pos, pacman_pos):
    l = []
    dir_array = np.array(pt_pos) - np.array(pacman_pos)
    if dir_array[0] > 0:
        l.append("right")
    elif dir_array[0] < 0:
        l.append("left")
    else:
        pass
    if dir_array[1] > 0:
        l.append("down")
    elif dir_array[1] < 0:
        l.append("up")
    else:
        pass
    return l

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
    dict_locs_df[(0, 18)][(1, 18)] = 1
    dict_locs_df[(29, 18)][(0, 18)] = 1
    dict_locs_df[(29, 18)][(28, 18)] = 1
    return dict_locs_df





# =====================================================

def add_states(df_reset):
    df_tmp = pd.DataFrame(
        [
            [np.nan] * 6 if isinstance(i, float) else i
            # for i in df_reset.contribution.to_list()
            for i in df_reset.weight.to_list()
        ],
        columns=["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"],
    )

    vague_mask = (
        np.sort(df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).values)[:, -1]
        - np.sort(df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).values)[:, -2]
    ) <= 0.1

    nan_mask = df_tmp.fillna(0).sum(1) == 0

    return pd.concat(
        [
            df_reset,
            pd.Series(
                [
                    ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"][i]
                    for i in df_tmp.values.argsort()[:, -1]
                ]
            )
            .mask(vague_mask)
            .fillna("vague")
            .mask(nan_mask)
            .rename("labels"),
            df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).add_suffix("_weight"),
        ],
        1,
    )


def consecutive_groups(df_total, state):
    trial_index = df_total.groupby("file").apply(lambda x: x.index.to_list()).values
    sel_index = []
    for trial in trial_index:
        trial_data = df_total.loc[trial]
        temp = [list(each) for each in cg(trial_data[trial_data["labels"] == state].index)]
        if len(temp) > 0:
            sel_index.extend(temp)
    return sel_index


def _vague(val):
    val = np.argsort(val)
    if np.all(val[-2:] == np.array([0, 1])) or np.all(val[-2:] == np.array([1, 0])) or np.all(val[-2:] == np.array([5, 1])):
        return False
    else:
        return True


def vague_consecutive_groups(df_total):
    vague_df = (
        df_total[df_total.labels == "vague"]
    )
    vague_df["useful_vague"] = vague_df.apply(lambda x: _vague(x.filter(regex="_weight").values), axis = 1)
    trial_index = vague_df.groupby("file").apply(lambda x: x.index.to_list()).values
    sel_index = []
    for trial in trial_index:
        trial_data = vague_df.loc[trial]
        temp = [list(each) for each in cg(trial_data[trial_data["useful_vague"] == 1].index)]
        if len(temp) > 0:
            sel_index.extend(temp)
    return sel_index


def evade_consecutive_groups(df_total):
    trial_index = df_total.groupby("file").apply(lambda x: x.index.to_list()).values
    sel_index = []
    for trial in trial_index:
        trial_data = df_total.loc[trial]
        temp = [list(each) for each in
                cg(trial_data[(trial_data["labels"] == "evade_blinky") | (trial_data["labels"] == "evade_clyde")].index)]
        if len(temp) > 0:
            sel_index.extend(temp)
    return sel_index


def saccDiff(df):
    is_sacc = np.where(df.values > 0)[0]
    if len(is_sacc) <= 1:
        return np.nan
    else:
        return np.diff(is_sacc)


def how_many_turns(df, start, end):
    dirs = df.loc[start:end, "pacman_dir_fill"]
    return ((dirs != dirs.shift()) & (dirs != dirs.replace(OPPOSITE_DIRS))).sum() - 1


def how_many_turns_poss(poss):
    # print()
    dirs = [
        relative_dir(poss[i + 1], poss[i])[0] if poss[i + 1] != poss[i] else np.nan
        for i in range(len(poss) - 1)
    ]
    dirs = pd.Series(dirs).fillna(method="ffill")
    return ((dirs != dirs.shift()) & (dirs != dirs.replace(OPPOSITE_DIRS))).sum() - 1


def add_dis(df_total, col1, col2, rename=None):
    """
    对任意两点添加距离
    """
    try:
        df_total = df_total.merge(
            LOCS_DF[["pos1", "pos2", "dis"]],
            left_on=[col1, col2],
            right_on=["pos1", "pos2"],
            how="left",
        ).drop(columns=["pos1", "pos2"])
        df_total.loc[(~df_total[col1].isnull()) & (~df_total[col2].isnull()) & (df_total['dis'].isnull()),'dis'] = 0
    except:
        df_total["dis"] = np.nan
    if rename:
        df_total = df_total.rename(columns={'dis':rename})
    return df_total


def add_PEG_dis(df_total):
    diss = add_dis(
        add_dis(
            add_dis(
                df_total[
                    ["ghost2Pos", "ghost1Pos", "next_eat_energizer", "pacmanPos"]
                ].reset_index(),
                "pacmanPos",
                "next_eat_energizer",
                "PE_dis",
            ),
            "next_eat_energizer",
            "ghost1Pos",
            "EG1_dis",
        ),
        "next_eat_energizer",
        "ghost2Pos",
        "EG2_dis",
    )

    diss["EG_dis"] = diss[["EG1_dis", "EG2_dis"]].min(1)
    df_total = pd.concat(
        [
            df_total,
            diss.set_index("index")[["PE_dis", "EG_dis", "EG1_dis", "EG2_dis"]],
        ],
        1,
    )
    return df_total


def _centralization(x, col):
    mean_val = np.nanmean(x[col].values)
    tmp = x[col].apply(lambda x: np.nan if np.isnan(x) else x-mean_val)
    # tmp[~np.isnan(tmp)] = tmp[~np.isnan(tmp)] - mean_val
    return tmp


def _centralization2(x, col, normalization):
    mean_val = np.nanmean(x[col].values)
    std_val = np.nanstd(x[col].values)
    tmp = x[col].apply(lambda x: (x - mean_val)/std_val if normalization else (x-mean_val))
    # tmp[~np.isnan(tmp)] = tmp[~np.isnan(tmp)] - mean_val
    return tmp


def _exclude(d, coef):
    mean_val = np.nanmean(d)
    std_val = np.nanstd(d)
    return d.apply(lambda x: np.nan if np.abs(x - mean_val) > coef * std_val or np.isnan(x) else x)


def z_score_wo_outlier2(df, col, coef, normalization):
    '''
    先去除outlier，再做centralization.
    '''
    new = col + "_z"
    # df[new] = df[col].apply(lambda x: x if x != 0 else np.nan)
    df[new] = df[col]
    tmp = df[["file", new]].groupby("file").apply(
        # lambda x: _centralization(x, new)
        lambda x: _exclude(x[new], coef)
    )
    df[new] = tmp.values.T if tmp.values.shape[0] != df.shape[0] else tmp.values

    # mean_val = np.nanmean(df[new])
    # std_val = np.nanstd(df[new])
    # df[new] = df[new].apply(
    #     lambda x: np.nan if np.abs(x - mean_val) > coef * std_val or np.isnan(x) else x
    # )
    # 利用整个trial的非空数据中心化
    tmp = df[["file", new]].groupby("file").apply(
        # lambda x: _centralization(x, new)
        lambda x: _centralization2(x, new, normalization)
    )
    df[new] = tmp.values.T if tmp.values.shape[0] != df.shape[0] else tmp.values
    return df


def pupil_consecutive_groups(df_total, state):
    trial_index = df_total.groupby("file").apply(lambda x: x.index.to_list()).values
    sel_index = []
    vague_trial_index = []
    for trial in trial_index:
        trial_data = df_total.loc[trial]
        temp = [list(each) for each in cg(trial_data[trial_data["labels"] == state].index)]
        if len(temp) > 0:
            sel_index.extend(temp)
            for _ in range(len(temp)):
                vague_trial_index.append([trial[0], trial[-1]])
    return sel_index, vague_trial_index


def seq_center(seq):
    return seq[len(seq) // 2]


# ======================================================

def generate_planned_accidental(df_total):
    df_total = df_total.reset_index(drop = True)

    # if overlap == False:
    #     labels = df_total.labels
    #     labels = labels.values[:, 0]
    #     df_total = df_total.drop(columns = ["labels"])
    #     df_total["labels"] = labels

    energizer_start_index = df_total[
        (df_total.eat_energizer == True)
        & (df_total[["ifscared1", "ifscared2"]].min(1).shift() < 3)
        ][["next_eat_rwd", "energizers", "ifscared1", "ifscared2"]].index
    energizer_lists = [
            (df_total.loc[i:, ["ifscared1", "ifscared2"]] <= 3)
                .max(1)
                .where(lambda x: x == True)
                .dropna()
                .index
        for i in energizer_start_index
    ]
    energizer_lists = [
        np.arange(i, energizer_lists[idx][0]) for idx, i in enumerate(energizer_start_index)
        if len(energizer_lists[idx]) > 0
    ]
    # energizer_lists = [
    #     np.arange(
    #         i,
    #         (df_total.loc[i:, ["ifscared1", "ifscared2"]] <= 3)
    #             .max(1)
    #             .where(lambda x: x == True)
    #             .dropna()
    #             .index[0],
    #     )
    #     for i in energizer_start_index
    # ]
    reindex_max = max([each[0] for each in energizer_lists if len(each) > 0])
    print("Max reindex : ", reindex_max)

    df_temp = (
        pd.Series(energizer_lists)
            .apply(lambda x: x[0] if len(x) > 0 else np.nan)
            .dropna()
            .reset_index()
            .astype(int)
            .set_index(0)
            # .reindex(range(417238))
            .reindex(range(reindex_max + 1))
            .rename(columns={"index": "last_index"})
    )
    df_temp.loc[~df_temp.last_index.isnull(), "last_index"] = df_temp.loc[
        ~df_temp.last_index.isnull()
    ].index
    df_temp = df_temp.fillna(method="bfill")
    pre_index = (
        add_dis(
            df_temp.reset_index()
                .rename(columns={0: "prev_index"})
                .merge(
                df_total["pacmanPos"].reset_index(),
                left_on="prev_index",
                right_on="index",
                how="left",
            )
                .drop(columns="index")
                .merge(
                df_total["pacmanPos"].reset_index(),
                left_on="last_index",
                right_on="index",
                how="left",
                suffixes=["_prev", "_last"],
            ),
            "pacmanPos_prev",
            "pacmanPos_last",
        )
            .sort_values(by="prev_index")
            .groupby("index")
            .apply(
            lambda x: x.set_index("prev_index")
                .dis.diff()
                .where(lambda x: x > 0)
                .dropna()
                .index.values[-1]
            if len(x.dis.diff().where(lambda x: x > 0).dropna()) > 0
            else x.prev_index.values[0]
        )
    )
    print("pre_index")

    # ====================
    #
    planned_traj_index = (
        pd.Series(energizer_lists).explode().rename("sec_level_1").reset_index()
            .merge(df_total.labels.reset_index(), left_on="sec_level_1", right_on="index")
    ).groupby("index_x").apply(
        # lambda x: int(x.index_x[0]) if (x.iloc[:10].labels == "approach").mean() > 0.8 and len(x) > 0 else None
        lambda x: int(x.index_x.values[0]) if (x.iloc[:10].labels == "approach").mean() > 0.8 and len(x) > 0 else None
    )
    planned_traj_index = [int(i) for i in planned_traj_index.values[~pd.isna(planned_traj_index.values)]]
    planned_lists = [np.arange(pre_index[energizer_lists[i][0]], energizer_lists[i][0]) for i in planned_traj_index]

    accidental_traj_index = (
        pd.Series(energizer_lists).explode().rename("sec_level_1").reset_index()
            .merge(df_total.labels.reset_index(), left_on="sec_level_1", right_on="index")
    ).groupby("index_x").apply(
        # lambda x: int(x.index_x[0]) if (x.iloc[:10].labels == "approach").mean() <=0.2 and (x.iloc[:10].labels == "local").mean() > 0.5 else None
        lambda x: int(x.index_x.values[0]) if (x.iloc[:10].labels == "approach").mean() <=0.2 and (x.iloc[:10].labels == "local").mean() > 0.5 else None
    )
    accidental_traj_index = [int(i) for i in accidental_traj_index.values[~pd.isna(accidental_traj_index.values)]]
    accidental_lists = [np.arange(pre_index[energizer_lists[i][0]], energizer_lists[i][0]) for i in accidental_traj_index]
    # ====================


    # ====================
    #
    planned_all = []
    accidental_all = []
    for i, each in enumerate(planned_lists):
        temp = list(copy.deepcopy(each))
        index = temp[-1] + 1
        while df_total.iloc[index].ifscared1 > 3 and df_total.iloc[index].ifscared2 > 3:
            temp.append(index)
            index += 1
        if df_total.iloc[index].ifscared1 == 3 or df_total.iloc[index].ifscared2 == 3:
            planned_all.append(copy.deepcopy(temp))
    for i, each in enumerate(accidental_lists):
        temp = list(copy.deepcopy(each))
        index = temp[-1] + 1
        while df_total.iloc[index].ifscared1 > 3 and df_total.iloc[index].ifscared2 > 3:
            temp.append(index)
            index += 1
        if df_total.iloc[index].ifscared1 == 3 or df_total.iloc[index].ifscared2 == 3:
            accidental_all.append(copy.deepcopy(temp))

    planned_all_start = [each[0] for each in planned_all]
    planeed_rest_list = []
    for i in planned_lists:
        if i[0] in planned_all_start:
            planeed_rest_list.append(i)
    planned_lists = planeed_rest_list
    return planned_lists, accidental_lists, planned_all, accidental_all


def generate_suicide_normal_next(df_total, suicide):
    df_total = df_total.reset_index(drop=True)
    # Drop duplicate "labels"
    # if overlap == False:
    #     labels = df_total.labels
    #     labels = labels.values[:, 0]
    #     df_total = df_total.drop(columns=["labels"])
    #     df_total["labels"] = labels
    select_last_num = 10
    if suicide == "normal":
        suicide_trial = df_total.groupby("file").apply(
            lambda x: [x.iloc[0].file, x.index[-select_last_num:], "suicide"]
            if np.sum(x.iloc[-select_last_num:].labels == "approach") == select_last_num
            # if np.sum(x.iloc[-select_last_num:].labels == "approach") >= 1
            #    and np.max(x.iloc[-select_last_num:][["ifscared1", "ifscared2"]].values) < 4
            else [x.iloc[0].file, x.index[-select_last_num:], "normal"]
        )
    elif suicide == "hard":
        suicide_trial = df_total.groupby("file").apply(
            lambda x: [x.iloc[0].file, x.index[-select_last_num:], "suicide"]
            if np.sum(x.iloc[-select_last_num:].labels == "approach") == select_last_num
            # if np.sum(x.iloc[-select_last_num:].labels == "approach") >= 1
            and np.max(x.iloc[-select_last_num:][["ifscared1", "ifscared2"]].values) < 4
            else [x.iloc[0].file, x.index[-select_last_num:], "normal"]
        )
    elif suicide == "easy":
        suicide_trial = df_total.groupby("file").apply(
            lambda x: [x.iloc[0].file, x.index[-select_last_num:], "suicide"]
            # if np.sum(x.iloc[-select_last_num:].labels == "approach") == select_last_num
            if np.sum(x.iloc[-select_last_num:].labels == "approach") >= 1
            #    and np.max(x.iloc[-select_last_num:][["ifscared1", "ifscared2"]].values) < 4
            else [x.iloc[0].file, x.index[-select_last_num:], "normal"]
        )
    elif suicide == "contribution":
        suicide_trial = df_total.groupby("file").apply(
            lambda x: [x.iloc[0].file, x.index[-select_last_num:], "suicide"]
            # if np.sum(x.iloc[-select_last_num:].labels == "approach") == select_last_num
            if np.all(
                np.vstack(x.iloc[-select_last_num:].contribution.values)[:,4] >= 0.2
            )
               and np.all(
                np.vstack(x.iloc[-select_last_num:].contribution.values)[:,[2,3]].reshape(-1) == 0.0
            )
            #    and np.all(
            #     np.vstack(x.iloc[-select_last_num:].contribution.values)[:, 0] <= 0.1
            # )
               and np.max(x.iloc[-select_last_num:][["ifscared1", "ifscared2"]].values) < 4
            else [x.iloc[0].file, x.index[-select_last_num:], "normal"]
        )
    print("Finsihed labelling.")

    """Expand the index in suicide_trial"""
    temp = (
        suicide_trial.apply(pd.Series, index=["file", "indexes", "list_status"])
        .drop("file", 1)
        .reset_index()
        .explode("indexes")
        .reset_index()
        .rename(columns={"index": "group_id"})
    )

    """Paste the label corresponding to each index"""
    temp = temp.merge(
        df_total[["labels", "contribution"]].reset_index(), left_on="indexes", right_on="index", how="left"
    ).drop("index", 1)

    """In order to distinguish the list in normal"""
    if suicide == "contribution":
        second_label = (
            temp.assign(
                # is_evade=temp.contribution.apply(lambda x: np.any(x[[2,3]] >= 0.5)),
                is_evade=temp.contribution.apply(lambda x: np.any(x[[2,3]] >= 0.2)),
                not_approach=temp.contribution.apply(lambda x: x[4] == 0.0),
                # is_local = temp.contribution.apply(lambda x: np.any(x[1] >= 0.2))
                # not_approach=temp.labels.apply(lambda x: True),

                # is_evade=temp.labels.str.contains("evade"),
                # not_approach=temp.contribution.apply(lambda x: x[4] == 0.0),
            )
                .groupby("group_id")
                .apply(
                lambda x: "normal"
                # if (x.is_evade.mean() == 1 or x.is_local.mean()==1) and x.not_approach.mean() == 1
                if x.is_evade.mean() == 1 and x.not_approach.mean() == 1
                else "others"
            )
        )
    else:
        second_label = (
            temp.assign(
                is_evade=temp.labels.str.contains("evade"),
                not_approach=temp.labels != "approach",
            )
                .groupby("group_id")
                .apply(
                lambda x: "normal"
                if x.is_evade.mean() == 1 and x.not_approach.mean() == 1
                else "others"
            )
        )
    """Classified into three lists"""
    lists = (
        temp.merge(
            second_label.rename("label2").reset_index(), how="left", on="group_id"
        )
        .groupby(["list_status", "label2", "group_id"])
        .apply(lambda x: list(x["indexes"]))
        .reset_index()
    )
    """Three kinds of lists are separated"""
    suicide_lists = list(lists[lists.list_status == "suicide"][0].values)
    normal_lists = list(
        lists[(lists.list_status != "suicide") & (lists.label2 == "normal")][0].values
    )
    other_lists = list(
        lists[(lists.list_status != "suicide") & (lists.label2 != "normal")][0].values
    )
    print("Finished extracting trials.")

    temp = df_total.loc[[i[0] for i in suicide_lists]]
    suicide_next_list = (
        temp.assign(next_game_trial=(temp.game_trial.astype(int) + 1).astype(str))
        .merge(
            df_total[["game", "game_trial"]].reset_index(),
            left_on=["game", "next_game_trial"],
            right_on=["game", "game_trial"],
            how="left",
            suffixes=["", "_match"],
        )
        .groupby(["game", "game_trial"])
        .apply(
            lambda x: list(x["index"].astype(int))[:10]
            if len(x["index"]) >= 10
            else np.nan
        )
        .dropna()
        .values.tolist()
    )

    temp = df_total.loc[[i[0] for i in normal_lists]]
    normal_next_list = (
        temp.assign(next_game_trial=(temp.game_trial.astype(int) + 1).astype(str))
        .merge(
            df_total[["game", "game_trial"]].reset_index(),
            left_on=["game", "next_game_trial"],
            right_on=["game", "game_trial"],
            how="left",
            suffixes=["", "_match"],
        )
        .groupby(["game", "game_trial"])
        .apply(
            lambda x: list(x["index"].astype(int))[:10]
            if len(x["index"]) >= 10
            else np.nan
        )
        .dropna()
        .values.tolist()
    )

    temp = df_total.loc[[i[0] for i in other_lists]]
    other_next_list = (
        temp.assign(next_game_trial=(temp.game_trial.astype(int) + 1).astype(str))
        .merge(
            df_total[["game", "game_trial"]].reset_index(),
            left_on=["game", "next_game_trial"],
            right_on=["game", "game_trial"],
            how="left",
            suffixes=["", "_match"],
        )
        .groupby(["game", "game_trial"])
        .apply(
            lambda x: list(x["index"].astype(int))[:10]
            if len(x["index"]) >= 10
            else np.nan
        )
        .dropna()
        .values.tolist()
    )

    return (
        suicide_lists,
        normal_lists,
        other_lists,
        suicide_next_list,
        normal_next_list,
        other_next_list,
    )



# =======
def largest_2ndlargest_diff(df):
    a = df.values
    # a = df.values[:, 2:] #TODO:只针对human数据
    return np.sort(a, axis=1)[:, -1] - np.sort(a, axis=1)[:, -2]


def extend_df_overlap(df_total, condition, df_overlap):
    df_filter = df_total.loc[
        condition,
        ["file", "index", "next_pacman_dir_fill", "pacmanPos", "pacman_dir_fill",],
    ]
    df_overlap = (
        df_overlap.assign(
            local_4dirs_diff=largest_2ndlargest_diff(df_overlap),
            largest_dir=df_overlap.eq(df_overlap.max(1), axis=0)
            .stack()
            .replace({False: np.nan})
            .dropna()
            .reset_index()
            .groupby(["file", "index"])
            .apply(lambda x: list(x.local_feature_dir)),
        )
        .reset_index()
        .merge(df_filter, on=["file", "index"], how="left",)
        .merge(MAP_INFO[["NextNum", "pos"]], left_on="pacmanPos", right_on="pos")
        .drop(columns=["pos"])
    )
    return df_overlap




def go_to_most_beans(
    df_overlap, cate_df, save_path, only_cross_fork, exclude_2dirs, landscape
):
    pp = []
    for n in [5]:
        if only_cross_fork:
            """这个仅仅适合n=5的情况"""
            # extend_cross_fork(df_overlap, save_path) #
            break
        if exclude_2dirs:
            df_overlap = exclude_2dirs(df_overlap)

        """1)是否选择了最大的方向choice_large 2) 这个位置是不是转弯口"""
        df_overlap = df_overlap.assign(
            choice_large=df_overlap.apply(
                lambda x: x.next_pacman_dir_fill == random.choice(x.largest_dir), 1
            ),
            if_cross=df_overlap.pacmanPos.isin(TURNING_POS),
        )

        """准备画图元素"""
        """如果需要按照地形来分的话，需要把comment掉的东西都恢复"""
        print()

        result_trial_df_list = []
        result_game_df_list = []
        result_day_df_list = []
        df_overlap["game"] = df_overlap.file.apply(lambda x: "-".join([x.split("-")[0]] + x.split("-")[2:6]))
        df_overlap["day"] = df_overlap.file.apply(lambda x: "-".join(x.split("-")[2:6]))
        for name, group in df_overlap.groupby("file"):
            tmp_res =  (
            group[group[["down", "up", "left", "right"]].max(1) > 0]
            .groupby(
                [
                    "NextNum",
                    "if_cross",
                    group[
                        group[["down", "up", "left", "right"]].max(1) > 0
                    ].local_4dirs_diff.apply(lambda x: min(x, 4)),
                ]
            )
            .choice_large.apply(
                lambda x: pd.Series({"mean": x.mean(), "count": len(x), "std": x.std()})
            )
            .unstack()
            .reset_index()
            ).merge(cate_df, on=["if_cross", "NextNum"], how="left",)
            result_trial_df_list.append(tmp_res)

        for name, group in df_overlap.groupby("game"):
            tmp_res =  (
            group[group[["down", "up", "left", "right"]].max(1) > 0]
            .groupby(
                [
                    "NextNum",
                    "if_cross",
                    group[
                        group[["down", "up", "left", "right"]].max(1) > 0
                    ].local_4dirs_diff.apply(lambda x: min(x, 4)),
                ]
            )
            .choice_large.apply(
                lambda x: pd.Series({"mean": x.mean(), "count": len(x), "std": x.std()})
            )
            .unstack()
            .reset_index()
            ).merge(cate_df, on=["if_cross", "NextNum"], how="left",)
            result_game_df_list.append(tmp_res)

        for name, group in df_overlap.groupby("day"):
            tmp_res =  (
            group[group[["down", "up", "left", "right"]].max(1) > 0]
            .groupby(
                [
                    "NextNum",
                    "if_cross",
                    group[
                        group[["down", "up", "left", "right"]].max(1) > 0
                    ].local_4dirs_diff.apply(lambda x: min(x, 4)),
                ]
            )
            .choice_large.apply(
                lambda x: pd.Series({"mean": x.mean(), "count": len(x), "std": x.std()})
            )
            .unstack()
            .reset_index()
            ).merge(cate_df, on=["if_cross", "NextNum"], how="left",)
            result_day_df_list.append(tmp_res)


        result_df = (
            df_overlap[df_overlap[["down", "up", "left", "right"]].max(1) > 0]
            .groupby(
                [
                    "NextNum",
                    "if_cross",
                    df_overlap[
                        df_overlap[["down", "up", "left", "right"]].max(1) > 0
                    ].local_4dirs_diff.apply(lambda x: min(x, 4)),
                ]
            )
            .choice_large.apply(
                lambda x: pd.Series({"mean": x.mean(), "count": len(x), "std": x.std()})
            )
            .unstack()
            .reset_index()
        ).merge(cate_df, on=["if_cross", "NextNum"], how="left",)
        #
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # if landscape:
        #     # plt.figure(dpi=300)
        #     fig, ax = plt.subplots(dpi=300)
        #     sns.scatterplot(
        #         data=result_df,
        #         x="local_4dirs_diff",
        #         y="mean",
        #         #             size=result_df["count"],
        #         ax=ax,
        #         hue="category",
        #         hue_order=["straight", "L-shape", "fork", "cross"],
        #         sizes=(20, 200),
        #     )
        #     for c in ["straight", "L-shape", "fork", "cross"]:
        #         gpd = result_df[result_df.category == c]
        #         ax.errorbar(
        #             gpd["local_4dirs_diff"],
        #             gpd["mean"],
        #             yerr=gpd["std"] / np.sqrt(gpd["count"]),
        #             marker=None,
        #             capsize=3,
        #         )
        #     ax.legend()
        #     ax.set_xticks([0, 1, 2, 3, 4])
        #     ax.set_xticklabels([0, 1, 2, 3, ">=4"])
        #     ax.set_xlabel("local reward max - 2nd max")
        #     ax.set_ylabel("% of toward the most valuable direction")
        #     ax.set_title("errorbar = traditional std")
        #     ax.set_ylim(0, 1)
        #     ax.figure.savefig(save_path)
        # else:
        #     errorplot = ax.errorbar(
        #         result_df[result_df.category == land]["local_4dirs_diff"],
        #         result_df[result_df.category == land]["mean"],
        #         #         yerr=result_df["std"] / np.sqrt(result_df["count"]),
        #         marker="o",
        #         #         capsize=3,
        #     )
        #     pp.append(errorplot.lines[0])
        #
        #     # use them in the legend
        #     ax.legend(pp, [str(i) for i in [1, 3, 5, 7]], ncol=4, numpoints=1)
        #     ax.set_xticks([0, 1, 2, 3, 4])
        #     ax.set_xticklabels([0, 1, 2, 3, ">=4"])
        #     plt.xlabel("local reward max - 2nd max")
        #     plt.ylabel("% of toward the most valuable direction")
        #     plt.title(land.capitalize() + " Local Graze")
        #     # plt.savefig(save_path)
        #     plt.show()
        return result_df, result_trial_df_list, result_game_df_list, result_day_df_list


def intersect_cnt(df, col1, col2):
    df["intersect_cnt"] = df.apply(
        lambda x: len(set(x[col1]) & set(x[col2]))
        if not isinstance(x[col1], float) and not isinstance(x[col2], float)
        else np.nan,
        1,
    )
    return df


def add_possible_dirs(df_total):
    for w in ["1", "2"]:
        df_total = (
            intersect_cnt(
                df_total.reset_index()
                .merge(
                    POSSIBLE_DIRS, left_on="pacmanPos", right_on="p_choice", how="left"
                )
                .set_index("level_0"),
                "level_1",
                "ghost" + w + "_wrt_pacman",
            )
            .sort_index()
            .rename(columns={"intersect_cnt": "intersect_cnt" + w})
        )
        df_total.loc[
            ~df_total["ghost" + w + "_wrt_pacman"].isnull(), "base" + w
        ] = df_total.loc[
            ~df_total["ghost" + w + "_wrt_pacman"].isnull(), "intersect_cnt" + w
        ] / df_total.loc[
            ~df_total["ghost" + w + "_wrt_pacman"].isnull(), "ghost" + w + "_wrt_pacman"
        ].map(
            len
        )
        df_total = df_total.drop(columns=["p_choice", "level_1", "intersect_cnt" + w])
    return df_total


def toward_ghost_table(df_total, cond=True):
    mapping_d = {"1": "red", "2": "yellow"}
    rs = pd.DataFrame()
    for w in ["1", "2"]:
        if "ghost" + w + "_dimi_manh" not in df_total.columns:
            df_total["ghost" + w + "_dimi_manh"] = (
                df_total[["next_pacman_dir_fill", "ghost" + w + "_wrt_pacman"]]
                .explode("ghost" + w + "_wrt_pacman")
                .apply(
                    lambda x: x["ghost" + w + "_wrt_pacman"]
                    == x["next_pacman_dir_fill"],
                    1,
                )
                .max(level=0)
            )
        rs = pd.concat(
            [
                rs,
                df_total[
                    (df_total.pacmanPos != df_total.pacmanPos.shift(-1))
                    & (df_total["ifscared" + w] != 3)
                    & (df_total["base" + w] == 0.5)
                    & (df_total["distance" + w] > 2)
                    & cond
                    #                     & (
                    #                         #                                                 df_total.index.isin(list(itertools.chain(*select_status[key])))
                    #                         df_total[select_status[key]]
                    #                         == 1
                    #                     )
                ]
                .groupby(
                    [
                        df_total["ifscared" + w] >= 3,
                        df_total["distance" + w].apply(lambda x: min(x, 25)),
                        "pacmanPos",
                        "ghost" + w + "Pos",
                    ]
                )["ghost" + w + "_dimi_manh"]
                .mean()
                .reset_index()
                .drop(columns=["pacmanPos", "ghost" + w + "Pos"])
                .groupby(["ifscared" + w, "distance" + w])["ghost" + w + "_dimi_manh"]
                .agg(["mean", "std", "count"])
                .unstack("ifscared" + w)
                .rename(
                    columns={
                        False: "ghost(" + mapping_d[w] + ") normal",
                        True: "ghost(" + mapping_d[w] + ") scared",
                    }
                ),
            ],
            1,
        )

    rs = rs.stack().reset_index()
    rs = pd.concat(
        [
            rs,
            rs.ifscared1.str.split(" ", expand=True).rename(
                columns={0: "ghost", 1: "status"}
            ),
        ],
        1,
    )
    all_rs = rs.copy()
    # -----
    result_trial_rs = []
    result_game_rs = []
    result_day_rs = []
    df_total["game"] = df_total.file.apply(lambda x: "-".join([x.split("-")[0]] + x.split("-")[2:6]))
    df_total["day"] = df_total.file.apply(lambda x: "-".join(x.split("-")[2:6]))
    for name, group in df_total.groupby("file"):
        rs = pd.DataFrame()
        for w in ["1", "2"]:
            if "ghost" + w + "_dimi_manh" not in group.columns:
                group["ghost" + w + "_dimi_manh"] = (
                    group[["next_pacman_dir_fill", "ghost" + w + "_wrt_pacman"]]
                        .explode("ghost" + w + "_wrt_pacman")
                        .apply(
                        lambda x: x["ghost" + w + "_wrt_pacman"]
                                  == x["next_pacman_dir_fill"],
                        1,
                    )
                        .max(level=0)
                )
            rs = pd.concat(
                [
                    rs,
                    group[
                        (group.pacmanPos != group.pacmanPos.shift(-1))
                        & (group["ifscared" + w] != 3)
                        & (group["base" + w] == 0.5)
                        & (group["distance" + w] > 2)
                        & cond
                        #                     & (
                        #                         #                                                 df_total.index.isin(list(itertools.chain(*select_status[key])))
                        #                         df_total[select_status[key]]
                        #                         == 1
                        #                     )
                        ]
                        .groupby(
                        [
                            group["ifscared" + w] >= 3,
                            group["distance" + w].apply(lambda x: min(x, 25)),
                            "pacmanPos",
                            "ghost" + w + "Pos",
                        ]
                    )["ghost" + w + "_dimi_manh"]
                        .mean()
                        .reset_index()
                        .drop(columns=["pacmanPos", "ghost" + w + "Pos"])
                        .groupby(["ifscared" + w, "distance" + w])["ghost" + w + "_dimi_manh"]
                        .agg(["mean", "std", "count"])
                        .unstack("ifscared" + w)
                        .rename(
                        columns={
                            False: "ghost(" + mapping_d[w] + ") normal",
                            True: "ghost(" + mapping_d[w] + ") scared",
                        }
                    ),
                ],
                1,
            )
        rs = rs.stack().reset_index()
        try:
            rs = pd.concat(
                [
                    rs,
                    rs.ifscared1.str.split(" ", expand=True).rename(
                        columns={0: "ghost", 1: "status"}
                    ),
                ],
                1,
            )
            result_trial_rs.append(rs.copy())
        except:
            print("Error data : {}".format(rs))
            continue
    # -----
    for name, group in df_total.groupby("game"):
        rs = pd.DataFrame()
        for w in ["1", "2"]:
            if "ghost" + w + "_dimi_manh" not in group.columns:
                group["ghost" + w + "_dimi_manh"] = (
                    group[["next_pacman_dir_fill", "ghost" + w + "_wrt_pacman"]]
                        .explode("ghost" + w + "_wrt_pacman")
                        .apply(
                        lambda x: x["ghost" + w + "_wrt_pacman"]
                                  == x["next_pacman_dir_fill"],
                        1,
                    )
                        .max(level=0)
                )
            rs = pd.concat(
                [
                    rs,
                    group[
                        (group.pacmanPos != group.pacmanPos.shift(-1))
                        & (group["ifscared" + w] != 3)
                        & (group["base" + w] == 0.5)
                        & (group["distance" + w] > 2)
                        & cond
                        #                     & (
                        #                         #                                                 df_total.index.isin(list(itertools.chain(*select_status[key])))
                        #                         df_total[select_status[key]]
                        #                         == 1
                        #                     )
                        ]
                        .groupby(
                        [
                            group["ifscared" + w] >= 3,
                            group["distance" + w].apply(lambda x: min(x, 25)),
                            "pacmanPos",
                            "ghost" + w + "Pos",
                        ]
                    )["ghost" + w + "_dimi_manh"]
                        .mean()
                        .reset_index()
                        .drop(columns=["pacmanPos", "ghost" + w + "Pos"])
                        .groupby(["ifscared" + w, "distance" + w])["ghost" + w + "_dimi_manh"]
                        .agg(["mean", "std", "count"])
                        .unstack("ifscared" + w)
                        .rename(
                        columns={
                            False: "ghost(" + mapping_d[w] + ") normal",
                            True: "ghost(" + mapping_d[w] + ") scared",
                        }
                    ),
                ],
                1,
            )

        rs = rs.stack().reset_index()
        try:
            rs = pd.concat(
                [
                    rs,
                    rs.ifscared1.str.split(" ", expand=True).rename(
                        columns={0: "ghost", 1: "status"}
                    ),
                ],
                1,
            )
            result_game_rs.append(rs.copy())
        except:
            print("Error data : {}".format(rs))
            continue
    # -----
    for name, group in df_total.groupby("day"):
        rs = pd.DataFrame()
        for w in ["1", "2"]:
            if "ghost" + w + "_dimi_manh" not in group.columns:
                group["ghost" + w + "_dimi_manh"] = (
                    group[["next_pacman_dir_fill", "ghost" + w + "_wrt_pacman"]]
                        .explode("ghost" + w + "_wrt_pacman")
                        .apply(
                        lambda x: x["ghost" + w + "_wrt_pacman"]
                                  == x["next_pacman_dir_fill"],
                        1,
                    )
                        .max(level=0)
                )
            rs = pd.concat(
                [
                    rs,
                    group[
                        (group.pacmanPos != group.pacmanPos.shift(-1))
                        & (group["ifscared" + w] != 3)
                        & (group["base" + w] == 0.5)
                        & (group["distance" + w] > 2)
                        & cond
                        #                     & (
                        #                         #                                                 df_total.index.isin(list(itertools.chain(*select_status[key])))
                        #                         df_total[select_status[key]]
                        #                         == 1
                        #                     )
                        ]
                        .groupby(
                        [
                            group["ifscared" + w] >= 3,
                            group["distance" + w].apply(lambda x: min(x, 25)),
                            "pacmanPos",
                            "ghost" + w + "Pos",
                        ]
                    )["ghost" + w + "_dimi_manh"]
                        .mean()
                        .reset_index()
                        .drop(columns=["pacmanPos", "ghost" + w + "Pos"])
                        .groupby(["ifscared" + w, "distance" + w])["ghost" + w + "_dimi_manh"]
                        .agg(["mean", "std", "count"])
                        .unstack("ifscared" + w)
                        .rename(
                        columns={
                            False: "ghost(" + mapping_d[w] + ") normal",
                            True: "ghost(" + mapping_d[w] + ") scared",
                        }
                    ),
                ],
                1,
            )

        rs = rs.stack().reset_index()
        try:
            rs = pd.concat(
                [
                    rs,
                    rs.ifscared1.str.split(" ", expand=True).rename(
                        columns={0: "ghost", 1: "status"}
                    ),
                ],
                1,
            )
            result_day_rs.append(rs.copy())
        except:
            print("Error data : {}".format(rs))
            continue
    return all_rs, result_trial_rs, result_game_rs, result_day_rs