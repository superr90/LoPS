import numpy as np
import copy
from copy import deepcopy
import pandas as pd
from itertools import groupby


def strategyFromWeight(weight, isStay, isVague, strategyToNumber, fileName, trailName):
    """
    Get the strategy according to weight
    """
    if isStay == True:
        return strategyToNumber["stay"]
    if isVague == True or np.sum(weight) == 0:
        return strategyToNumber["vague"]
    try:
        MIN = np.min(weight)
        if MIN < 0:
            weight = weight - MIN
        weight = weight / np.sum(weight)
        weight = list(weight)
        MAX = np.max(weight)
        index = np.where(weight == MAX)[0]
        if len(index) > 1:
            if strategyToNumber["local"] in index:
                return strategyToNumber["local"]
            elif strategyToNumber["global"] in index:
                return strategyToNumber["global"]
            elif strategyToNumber["global"] not in index and strategyToNumber["local"] not in index and \
                    strategyToNumber["energizer"] not in index and strategyToNumber["approach"] not in index and \
                    strategyToNumber["no_energizer"] not in index:
                return index[0]
            return strategyToNumber["vague"]
        else:
            return index[0]
    except:
        print("====" * 30)
        print(fileName, trailName)
        return index[0]


def revise_function(data, context, revise_weight, main_agent, agents, agents_list, agent_num, suffix):
    """
    data:data
    context:Paragraphs that need to be modified
    revise_weight：modified weight
    main_agnet:The agent to be judged
    """
    for prev, end in context:
        all_data = copy.deepcopy(data.loc[prev:end - 1])
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            # print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind1 = np.where(temp_data.nan_dir == False)[0] + prev
        ind2 = np.where(temp_data.nan_dir == True)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
        num_samples = all_data.shape[0]
        pre_estimation = all_data[agents_list].values
        agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
        for each_sample in range(num_samples):
            for each_agent in range(len(agents_list)):
                agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                    each_agent
                ]
        agent_accuracy = []
        for i in range(agent_num):
            accuracy = 0
            dir_Q_value = agent_Q_value[:, :, i]
            for each_sample in range(num_samples):
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
            agent_accuracy.append(accuracy / num_samples)

        main_agent_accuracy = agent_accuracy[main_agent]
        max_accuracy = np.max(agent_accuracy)
        if main_agent_accuracy / max_accuracy > 0.8 and main_agent_accuracy > 0.6:
            # print(prev, end)
            data["revise_weight"].loc[prev:end - 1] = [revise_weight] * (end - prev)

            phase_is_correct, estimated_dir, rate = _calculate_is_correct(np.array(revise_weight), all_data, true_prob,
                                                                          agents,
                                                                          suffix=suffix)
            data["revise_is_correct"].loc[ind1] = np.array(phase_is_correct, dtype=np.int)
            data["revise_is_correct"].loc[ind2] = [np.nan] * len(ind2)
            data["predict_dir"].loc[ind1] = np.array(estimated_dir)
            data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)


def revise_vague(data, context, agents, agents_list, agent_num, suffix, strategyToNumber):
    """
    data:data
    context:Paragraphs that need to be modified
    revise_weight：modified weight
    main_agnet:The agent to be judged
    """
    for prev, end in context:
        all_data = copy.deepcopy(data.loc[prev:end - 1])
        weight = all_data["revise_weight"].iloc[0]
        if np.sum(weight) <= 0:
            continue
        else:
            max_index = np.where(weight == np.max(weight))[0]
            if len(max_index) == 1:
                maxIndex = np.argmax(weight)
                if "ifscared3" in all_data.columns and all_data["ifscared3"].iloc[0] == -1 and (
                        maxIndex == strategyToNumber["evade_3"] or maxIndex == strategyToNumber["evade_4"]):
                    continue
                data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)
                continue
        all_data = copy.deepcopy(data.loc[prev:end - 1])
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            # print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind1 = np.where(temp_data.nan_dir == False)[0] + prev
        ind2 = np.where(temp_data.nan_dir == True)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
        num_samples = all_data.shape[0]
        pre_estimation = all_data[agents_list].values
        agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
        for each_sample in range(num_samples):
            for each_agent in range(len(agents_list)):
                agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                    each_agent
                ]
        agent_accuracy = []

        weight = all_data["revise_weight"].iloc[0]
        max_index = np.where(weight == np.max(weight))[0]

        max_accuracy_index = -1
        max_accuracy = -1
        for i in max_index:
            accuracy = 0
            dir_Q_value = agent_Q_value[:, :, i]
            for each_sample in range(num_samples):
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
            if accuracy / num_samples >= max_accuracy:
                max_accuracy_index = i
                max_accuracy = accuracy / num_samples

        revise_weight = [0] * agent_num
        revise_weight[max_accuracy_index] = 1

        temp = 4 - np.sum(np.isinf(agent_Q_value[0, :, 0]))
        if max_accuracy > 1 / temp:
            # print(prev, end)
            data["revise_weight"].loc[prev:end - 1] = [revise_weight] * (end - prev)

            phase_is_correct, estimated_dir, rate = _calculate_is_correct(np.array(revise_weight), all_data, true_prob,
                                                                          agents,
                                                                          suffix=suffix)
            data["revise_is_correct"].loc[ind1] = np.array(phase_is_correct, dtype=np.int)
            data["revise_is_correct"].loc[ind2] = [np.nan] * len(ind2)
            data["predict_dir"].loc[ind1] = np.array(estimated_dir)
            data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)


def reviseWrongEnergizer(data, energizerContext, strategyToNumber, agents, agents_list, suffix):
    """
    Modify paragraphs that were incorrectly marked as energizers
    :param data:
    :param energizerContext:
    :param strategyToNumber:
    :param agents:
    :param agents_list:
    :param suffix:
    :return:
    """
    for context in energizerContext:
        prev = context[0]
        end = context[1] + 1

        all_data = copy.deepcopy(data.loc[prev:end - 1])
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            # print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind1 = np.where(temp_data.nan_dir == False)[0] + prev
        ind2 = np.where(temp_data.nan_dir == True)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)

        tempIndex = context[1] + 1
        if tempIndex > data["level_0"].iloc[-1]:
            continue
        if (data["strategy"].loc[tempIndex] != strategyToNumber["approach"] and data["strategy"].loc[tempIndex] ==
                strategyToNumber["local"]):  # or (
            # data["strategy"].loc[tempIndex] != strategyToNumber["approach"] and context[0] - 1 >
            # data["level_0"].iloc[0] and data["strategy"].loc[context[0] - 1] ==
            # strategyToNumber["local"]):

            orignalWeight = deepcopy(data["revise_weight"].loc[prev])
            revise_weight = [0] * len(agents_list)
            revise_weight[strategyToNumber["local"]] = 1
            data["revise_weight"].loc[prev:end - 1] = [revise_weight] * (end - prev)

            phase_is_correct, _, rate = _calculate_is_correct(np.array(revise_weight), all_data, true_prob, agents,
                                                              suffix=suffix)

            _, _, originalRate = _calculate_is_correct(np.array(orignalWeight), all_data, true_prob, agents,
                                                       suffix=suffix)
            if rate / originalRate < 0.8:
                data["revise_weight"].loc[prev:end - 1] = [orignalWeight] * (end - prev)
                continue
            data["strategy"].loc[prev:end - 1] = [strategyToNumber["local"]] * (end - prev)
            data["revise_is_correct"].loc[ind1] = np.array(phase_is_correct, dtype=np.int)
            data["revise_is_correct"].loc[ind2] = [np.nan] * len(ind2)
            data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)


def setWeight(data, context, agents, revise_weight, suffix):
    for prev, end in context:
        all_data = copy.deepcopy(data.loc[prev:end - 1])
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            # print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind1 = np.where(temp_data.nan_dir == False)[0] + prev
        ind2 = np.where(temp_data.nan_dir == True)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)

        data["revise_weight"].loc[prev:end - 1] = [revise_weight] * (end - prev)
        phase_is_correct, estimated_dir, rate = _calculate_is_correct(np.array(revise_weight), all_data, true_prob,
                                                                      agents,
                                                                      suffix=suffix)
        data["revise_is_correct"].loc[ind1] = np.array(phase_is_correct, dtype=np.int)
        data["revise_is_correct"].loc[ind2] = [np.nan] * len(ind2)
        data["predict_dir"].loc[ind1] = np.array(estimated_dir)
        data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)


def reviseApproach(data, context, agents, agents_list, agent_num, suffix, strategyToNumber):
    for prev, end in context:
        all_data = copy.deepcopy(data.loc[prev:end - 1])
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            # print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind1 = np.where(temp_data.nan_dir == False)[0] + prev
        ind2 = np.where(temp_data.nan_dir == True)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
        num_samples = all_data.shape[0]
        pre_estimation = all_data[agents_list].values
        agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
        for each_sample in range(num_samples):
            for each_agent in range(len(agents_list)):
                agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                    each_agent
                ]
        agent_accuracy = []

        weight = all_data["revise_weight"].iloc[0]
        max_index = np.where(weight == np.max(weight))[0]

        max_accuracy_index = -1
        max_accuracy = -1
        for i in list(range(len(agents))):
            accuracy = 0
            dir_Q_value = agent_Q_value[:, :, i]
            for each_sample in range(num_samples):
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
            if accuracy / num_samples >= max_accuracy:
                max_accuracy_index = i
                max_accuracy = accuracy / num_samples
            agent_accuracy.append(accuracy / num_samples)

        accuracyApproach = agent_accuracy[strategyToNumber["approach"]]
        agent_accuracy[strategyToNumber["approach"]] = 0
        maxIndex = np.argmax(agent_accuracy)

        if (accuracyApproach == 0 or agent_accuracy[maxIndex] / accuracyApproach > 0.8):
            revise_weight = [0] * len(agents_list)
            revise_weight[maxIndex] = 1
            # print(prev, end)
            data["revise_weight"].loc[prev:end - 1] = [revise_weight] * (end - prev)

            phase_is_correct, estimated_dir, rate = _calculate_is_correct(np.array(revise_weight), all_data, true_prob,
                                                                          agents,
                                                                          suffix=suffix)
            data["revise_is_correct"].loc[ind1] = np.array(phase_is_correct, dtype=np.int)
            data["revise_is_correct"].loc[ind2] = [np.nan] * len(ind2)
            data["predict_dir"].loc[ind1] = np.array(estimated_dir)
            data["is_vague"].loc[prev:end - 1] = [False] * (end - prev)


def reviseMain(path, savePath, strategy_number, agents, agents_list, agent_num, suffix, scared_time):
    df = pd.read_pickle(path)
    df["DayTrial"] = df.file


    df["revise_weight"] = copy.deepcopy(np.array(df["contribution"]))
    df["revise_is_correct"] = copy.deepcopy(np.array(df["is_correct"]))

    trial_name_list = np.unique(df.file.values)
    all_trial_record = []


    for t, trial_name in enumerate(trial_name_list):
        data = df[df.file == trial_name]
        if "ifscared3" in data.columns and data["ifscared3"].iloc[0] != -1:
            continue
        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)

        # Revise vague paragraph
        vague_index = np.where(data["is_vague"] == True)[0] + data["level_0"].iloc[0]
        vague_context = list(set(list(np.array(data["trial_context"].loc[vague_index]))))
        vague_context.sort(key=lambda x: x[0])
        revise_vague(data, vague_context, agents, agents_list, agent_num, suffix, strategy_number)

        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)

        # Revise the approach. If it is determined to be an approach and no ghosts have been eaten, modify it to another strategy based on probability
        context_approach = np.where(data["strategy"] == strategy_number["approach"])[0]
        context_approach = list(set(list(data["trial_context"].iloc[context_approach])))
        eat_ghost = np.where(data["eat_ghost"] == True)[0] - 1 + data["level_0"].iloc[0]  # 吃到energizer的前一位置
        for pre, end in deepcopy(context_approach):
            isEatGhost = [1 if e >= pre and e < end else 0 for e in eat_ghost]
            if 1 in isEatGhost:
                context_approach.remove((pre, end))
        reviseApproach(data, context_approach, agents, agents_list, agent_num, suffix, strategy_number)
        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)

        # Revise the weight of the energizer paragraph
        eat_energizer = np.where(data["eat_energizer"] == True)[0] - 1 + data["level_0"].iloc[0]  #  the previous position of eating energizer
        eat_energizer_context = list(np.array(data["trial_context"].loc[eat_energizer]))  # the paragraphs of eating energizer
        revise_weight = [0] * len(agents_list)
        revise_weight[strategy_number["energizer"]] = 1
        revise_function(data=data, context=eat_energizer_context, revise_weight=revise_weight,
                        main_agent=strategy_number["energizer"],
                        agents=agents, agents_list=agents_list, agent_num=agent_num, suffix=suffix)
        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)
        # Revise the weight of the paragraph next to the energizer paragraph.
        eat_energizer_next = []
        for prev, end in eat_energizer_context:
            eat_energizer_next.append(end)
        index = list(np.array(data.index))
        eat_energizer_next = [i for i in eat_energizer_next if i in index]
        eat_energizer_next_context = list(np.array(data["trial_context"].loc[eat_energizer_next]))
        # If there is no ghost-eating in this paragraph, it will not be changed.
        for pre, end in deepcopy(eat_energizer_next_context):
            IS1 = list(data.loc[pre:end]["ifscared1"])
            IS2 = list(data.loc[pre:end]["ifscared2"])
            if (3 not in IS1) and (3 not in IS2):
                if end + 1 < data.iloc[-1]["level_0"] and data.loc[end + 1]["strategy"] != strategy_number["approach"]:
                    eat_energizer_next_context.remove((pre, end))
                elif end + 1 > data.iloc[-1]["level_0"]:
                    eat_energizer_next_context.remove((pre, end))

        revise_weight = [0] * len(agents_list)
        revise_weight[strategy_number["approach"]] = 1
        revise_function(data=data, context=eat_energizer_next_context, revise_weight=revise_weight,
                        main_agent=strategy_number["approach"],
                        agents=agents, agents_list=agents_list, agent_num=agent_num, suffix=suffix)

        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)

        # If there are multiple approaches after eating an energizer, merge them
        for i, ea in enumerate(eat_energizer):
            pre = ea + 1
            if i < len(eat_energizer) - 1:
                end = eat_energizer[i + 1] + 1
            else:
                end = data["level_0"].iloc[-1] + 1
            index = np.where(data.loc[pre:end - 1]["strategy"] == strategy_number["approach"])[0]
            index_context = list(set(list(data.loc[pre:end - 1]["trial_context"].iloc[index])))
            # If a ghost has been eaten in a paragraph
            index_context.sort(key=lambda x: x[0])
            if len(index_context) <= 1:
                continue
            # print("===================" * 100)
            new_context = [index_context[0]]
            for context in index_context[1:]:
                if context[0] - new_context[-1][0] <= scared_time:
                    new_context[-1] = (new_context[-1][0], context[1])
                else:
                    new_context.append(context)
            revise_weight = [0] * len(agents_list)
            revise_weight[strategy_number["approach"]] = 1
            setWeight(data, new_context, agents, revise_weight, suffix)

        data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
            lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
            axis=1)

        # Revise the paragraph that mistakenly marked local as energizer
        energizerIndex = np.where(data["strategy"] == strategy_number["energizer"])[0] + data["level_0"].iloc[
            0]  # 吃到energizer的前一位置
        groups = groupby(enumerate(energizerIndex), lambda i_x: i_x[0] - i_x[1])

        energizerContext = [(g[0][1], g[-1][1]) for _, g in ((k, list(group)) for k, group in groups)]
        reviseWrongEnergizer(data, energizerContext, strategy_number, agents, agents_list, suffix)

        all_trial_record.append(copy.deepcopy(data))

    df = pd.concat(all_trial_record)
    df.reset_index(inplace=True, drop=True)
    data["strategy"] = data[["revise_weight", "is_stay", "is_vague"]].apply(
        lambda x: strategyFromWeight(x.revise_weight, x.is_stay, x.is_vague, strategy_number, path, trial_name),
        axis=1)

    df.to_pickle(savePath + path.split("/")[-1])

    is_correct = df["revise_is_correct"]
    print(path, np.mean(is_correct))



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
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


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
    rate = 0
    for i, q in enumerate(estimated_prob):
        index = np.where(q == np.max(q))[0]
        if true_dir[i] in list(index):
            rate += 1 / len(index)
    rate /= len(estimated_prob)
    return is_correct, estimated_dir, rate


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
