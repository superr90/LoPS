import copy
import os
import pickle
import scipy
import numpy as np
import re
import pandas as pd
import math
import sys
import warnings

sys.path.append("../")
warnings.filterwarnings("ignore")
from PGM.bayesianScore import BDscore, learnBayesNet_noparallelize, learnBayesNetBlock
# from bayesianScore import learnBayesNet_Option, BDscore, learnBayesNet, learnBayesNet_noparallelize, learnBayesNetBlock
import random

from functools import partial
import multiprocessing

from copy import deepcopy
from PGM.PCalgorithm import PCskletetonData

from collections import Counter


# from scripts.human_data_process_no_energizer.states_fmri import PC
def PC(data, sampleNumber=4000):
    data_num = data.shape[1]
    if sampleNumber > data_num:
        sampleNumber = data_num
    index = np.random.choice(a=list(range(data_num)), p=[1 / data_num] * data_num, replace=False, size=sampleNumber)
    data = data[:, index]
    # data = pd.DataFrame(data.T)
    G, S = PCskletetonData(data)
    return G


class Simulator:
    def __init__(self, p0=0.8, p1=0.7, p2=0.6, p22=0.9, p3=0.8, p4=0.7):
        self.p0 = p0  # When the selected action has nothing to do with s0, the probability that s0 remains unchanged
        self.p1 = p1  # When the selected action has nothing to do with s1, the probability that s1 remains unchanged
        self.p2 = p2  # When the selected action has nothing to do with s2, the probability that s2 remains unchanged (the probability of the ghost converting from scared to normal)
        self.p22 = p22  # When the selected action has nothing to do with s2, the probability that s2 remains unchanged (the probability of the ghost converting from normal to scared)
        self.p3 = p3  # When the selected action has nothing to do with s3, the probability that s3 remains unchanged
        self.p4 = p4  # The probability of death when the chosen action is not e and the distance is closer to you

    def transition(self, state, action):
        # rule1:s0 (whether there are beans in close range) conversion rules
        if action == "L":
            s0 = 0
        elif action == "G":
            s0 = 1
        else:
            s0 = np.random.choice(a=[state[0], 1 - state[0]], p=[self.p0, 1 - self.p0], size=1)[0]

        # rule2:s1 (distance from ghost) conversion rule
        if action == "e" or (action == "A" and state[2] == 1):  #
            s1 = 1
        elif action == "A":
            s1 = 0
        else:
            s1 = np.random.choice(a=[state[1], 1 - state[1]], p=[self.p1, 1 - self.p1], size=1)[0]
        # rule3:s2 (ghost state rule) conversion rule
        if action == "E" and state[3] == 0:
            s2 = 1
        elif action == "A" and state[2] == 1:
            s2 = 0
        else:
            if state[2] == 1:
                s2 = np.random.choice(a=[state[2], 1 - state[2]], p=[self.p2, 1 - self.p2], size=1)[0]
            else:
                s2 = np.random.choice(a=[state[2], 1 - state[2]], p=[self.p22, 1 - self.p22], size=1)[0]
        # rule4:s4 ((distance from energizer) conversion rule
        if action == "E" and state[3] == 0:
            s3 = 1
        elif action == "E" and state[3] == 1:
            s3 = 0
        else:
            s3 = np.random.choice(a=[state[3], 1 - state[3]], p=[self.p3, 1 - self.p3], size=1)[0]
        # rule5:death rule
        if action != "e" and state[1] == 0:
            dead = np.random.choice(a=[True, False], p=[self.p4, 1 - self.p4], size=1)[0]
        else:
            dead = False

        if dead == True:
            return None
        else:
            return [s0, s1, s2, s3]

    def policy(self, state):
        pass

    def generator_from_simulation_envoroment(self, Type=1, sample_num=8000):
        init_states = [
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
        ]
        init_pro = [1 / len(init_states)] * len(init_states)
        state_sequence = []
        action_sequence = []
        real_pro = {}  # Real grammar appearance probability statistics
        length = 0
        actionList = []
        while length < sample_num:
            # 每一个trial
            states = []
            actions = []
            init_state = \
                np.random.choice(a=list(range(len(init_states))), p=init_pro, size=1)[
                    0]
            state = copy.deepcopy(init_states[init_state])
            while state is not None:
                self.strategy = self.policy(state)
                flag = True
                temp = ""
                temp_actions = []
                temp_states = []
                for i in range(len(self.strategy)):
                    action = self.strategy[i]
                    temp_states.append(copy.deepcopy(state))
                    temp_actions.append(action)
                    temp += action
                    state = self.transition(state, action)
                    if state is None:
                        flag = False
                        break
                if flag == True:
                    states += temp_states
                    actions += temp_actions
                    actionList.append(temp)
                    if real_pro.__contains__(temp):
                        real_pro[temp] += 1
                    else:
                        real_pro.update({temp: 1})
                if len(actions) == Type:
                    break
                # if length + len(actions) >= sample_num:
                #     break
            if len(actions) == Type:
                length += len(actions)
                state_sequence.append(states)
                action_sequence.append(actions)

        new_actions = []
        new_states = []
        for i in range(len(action_sequence)):
            new_actions += action_sequence[i]
            new_states += state_sequence[i]

        actions = new_actions
        states = new_states
        state = {
            "s0": [], "s1": [], "s2": [], "s3": [],
        }
        for i in range(len(states)):
            state["s0"].append(states[i][0])
            state["s1"].append(states[i][1])
            state["s2"].append(states[i][2])
            state["s3"].append(states[i][3])
        state = np.array(list(state.values()))
        state = pd.DataFrame(state.T, columns=["s0", "s1", "s2", "s3"], dtype=np.int64)
        S = ["G", "L", "e", "E", "A"]
        seq = "".join(actions)
        number = np.sum(list(real_pro.values()))
        for key in real_pro.keys():
            real_pro[key] /= number
        return seq, S, state, real_pro, actionList


class Simulator_grammar(Simulator):
    def __init__(self, policy_table, possibility, Type="one", p0=0.8, p1=0.7, p2=0.6, p22=0.9, p3=0.8, p4=0.7):
        super(Simulator_grammar, self).__init__(p0, p1, p2, p22, p3, p4)
        self.policy_table = policy_table
        self.possibility = possibility
        self.Type = Type

    def policy(self, state):
        T = 5
        state = [str(s) for s in state]
        state = "".join(state)
        pro = self.possibility[state]
        pro = np.exp(np.array(pro) * T)
        pro = pro / np.sum(pro)
        index = np.random.choice(a=list(range(len(pro))), p=pro, size=1)[0]
        return self.policy_table[state][index]


class Tools:
    def static_pro(self, seq, S):
        """
        Count the probability of each grammar in S appearing in seq
        :param seq:
        :param S:
        :return:
        """
        pro = {}
        for s in S:
            pro.update({s: 0})
        for s in seq:
            pro[s] += 1
        pro = np.array(list(pro.values())) / np.sum(list(pro.values()))

        return list(pro)

    def choice_max_n(self, ratios, chunks, component):
        """
       Select the n chunks with the largest ratio based on ratios
        :param ratios:
        :param chunks:
        :return:
        """
        # Sort by ratio from largest to smallest
        index = sorted(enumerate(ratios), key=lambda x: x[1], reverse=True)
        if len(index) == 0:
            return [], []
        index = [i[0] for i in index]
        ratios = np.array(ratios)[index]
        chunks = np.array(chunks)[index]
        component = np.array(component)[index]
        # Filter out those with ratio greater than 1
        index = np.where(ratios > 1)[0]
        if len(index) == 0:
            return [], [], []
        ratios = np.array(ratios)[index]
        chunks = np.array(chunks)[index]
        component = np.array(component)[index]
        # Filter out those whose difference from the maximum value is not greater than 0.02
        index = [0]
        for i in range(1, len(ratios)):
            if ratios[i] / ratios[0] > 0.85:
                index.append(i)
            else:
                break
        ratios = list(np.array(ratios)[index])
        chunks = list(np.array(chunks)[index])
        component = list(np.array(component)[index])
        return chunks, ratios, component

    def KL(self, P, Q):
        n = 0
        for key in P.keys():
            p = P[key]
            if Q.__contains__(key):
                q = Q[key]
            else:
                q = 0.00001
            n += (p * math.log2(p / q))
        return n


class Chunk:
    def __init__(self):
        self.tools = Tools()

    def parse(self, sequence, sets, place_sets, state=None):
        """
        Use the grammar in sets to divide the sequence to the greatest extent. After the division is completed, use the characters in place_sets to replace the divided paragraphs to generate a new sequence.
        :param sequence:
        :param sets:
        :param place_sets:
        :return:
        """
        cover = []
        sets = list(sets)
        pointer = 0
        while pointer < len(sequence):
            l = 0
            index = 0
            for i in range(len(sets)):
                L = len(sets[i])
                if sequence[pointer:pointer + L] == sets[i] and L > l:
                    l = L
                    index = i
            if l == 0:
                print("*" * 100)
            cover.append(index)
            pointer = pointer + l
        seq = ""
        for i in cover:
            seq += place_sets[i]
        if state is not None:
            new_state = []
            index = 0
            for i in cover:
                l = len(sets[i])
                new_state.append(list(state.iloc[index]))
                index += l
            new_state = np.array(new_state)
            state = pd.DataFrame(new_state, columns=state.columns)
        return seq, state

    def parse_pro(self, sequence, sets, remove_n=False):
        """
         Use the grammar in sets to divide the sequence to the greatest extent and count the probability of each grammar appearing.
        :param sets:
        :param place_sets:
        :return:
        """
        cover = []
        sets = list(sets)
        pointer = 0
        position_gram = []
        while pointer < len(sequence):
            l = 0
            index = 0
            for i in range(len(sets)):
                L = len(sets[i])
                if sequence[pointer:pointer + L] == sets[i] and L > l:
                    l = L
                    index = i
            if l == 0:
                print("=" * 100)
            cover.append(index)
            pointer = pointer + l
            position_gram += [sets[index]] * L
        pro = {}
        for s in sets:
            pro.update({s: 0})
        for i in cover:
            pro[sets[i]] += 1
        pro = np.array(list(pro.values()))
        frequence = deepcopy(pro)
        pro = pro / np.sum(pro)
        coverSet = [sets[c] for c in cover]
        return sets, pro, position_gram, frequence, coverSet

    def deep(self, t, aggregate_dict):
        """
        Convert each combination grammar represented by a code into the original (G, L, e, A, E). If 0 is EA, convert EA, and 2 is 0G, then convert 2 to EAG.
        :param t: 
        :param aggregate_dict: 
        :return: 
        """
        flag = True
        for i in range(len(t)):
            if aggregate_dict.__contains__(t[i]):
                flag = False
                break
        if flag == True:
            return t
        temp = ""
        for i in range(len(t)):
            if aggregate_dict.__contains__(t[i]):
                temp += self.deep(aggregate_dict[t[i]], aggregate_dict)
            else:
                temp += t[i]
        return temp

    def get_cover_set(self, S, aggregate_dict, gramLen):
        """
        Convert all the grammar represented by the current codename to the original representation
        :param S: 
        :param aggregate_dict: 
        :return: 
        """
        new_s = []
        Len = []
        for s in S:
            new_s.append(self.deep(s, aggregate_dict))
            Len.append(gramLen[s])
        sets = []
        L = []
        for i in range(len(new_s)):
            if new_s[i] not in sets:
                sets.append(new_s[i])
                L.append(Len[i])
        return sets, L

    def organize_data(self, seq, S, state=None, condition=None):
        """
        sequence data is organized into Dataframe format
        :param seq: string such as: 'GLeA'
        :param S: set
        :param state:DataFrame
        :return: DataFrame
        """
        if state is not None:
            state.reset_index(inplace=True, drop=True)
        data_parent = {}
        for s in S:
            data_parent.update({s: np.ones(len(seq) - 1)})
        data_child = {}
        for s in S:
            data_child.update({s: np.ones(len(seq) - 1)})
        if state is not None:
            data_condition = {}
            data_policy_condition = {}
            for s in state.columns:
                data_condition.update({s: np.ones(len(seq) - 1)})
                data_policy_condition.update({s: np.ones(len(seq) - 1)})
        else:
            data_condition = None
            data_policy_condition = None

        for i in range(1, len(seq)):
            data_parent[seq[i - 1]][i - 1] = 2
            data_child[seq[i]][i - 1] = 2
            if state is not None:
                for s in state.columns:
                    data_condition[s][i - 1] = state[s].iloc[i] + 1
                    data_policy_condition[s][i - 1] = state[s].iloc[i - 1] + 1

        data_parent = pd.DataFrame(data_parent, dtype=np.int64)
        data_child = pd.DataFrame(data_child, dtype=np.int64)
        if state is not None:
            data_condition = pd.DataFrame(data_condition, dtype=np.int64)
            data_policy_condition = pd.DataFrame(data_policy_condition, dtype=np.int64)
        condition_state = None
        if state is not None:
            # Get the state related to each strategy
            data = pd.concat([data_policy_condition, data_parent], axis=1).values.T
            data = np.array(data, dtype=np.int64)

            # num = data.shape[1]
            # index = np.random.choice(a=list(range(num)), p=[1 / num] * num, size=2800, replace=False)
            # data = data[:, index]

            nstates = np.max(data, axis=1).T
            nstates = np.array(nstates, dtype=np.int64)
            casual_num = data_policy_condition.shape[1]
            effect_num = data_parent.shape[1]
            blockMessage = {i: [i] for i in range(casual_num)}

            Alearn, parameters, bestparents, bestscores = learnBayesNetBlock(data=data, nstates=nstates,
                                                                             blockMessage=blockMessage,
                                                                             casualNum=casual_num,
                                                                             blockNum=len(blockMessage),
                                                                             effectNum=effect_num, conditions=condition)
            # print("condition_state completed!")
            condition_state = []
            names = np.array(list(data_condition.columns))
            for i in range(casual_num, casual_num + effect_num):
                index = np.where(Alearn[:, i] == 1)[0]
                index = list(names[index])
                condition_state.append(index)
        # print(condition_state)
        return data_child, data_parent, data_condition, condition_state

    def organize_data_skip_gram(self, seq, S, state=None):
        """
        sequence data is organized into Dataframe format
        :param seq: string such as: 'GLeA'
        :param S: set
        :param state:DataFrame
        :return: DataFrame
        """
        if state is not None:
            state.reset_index(inplace=True, drop=True)
        data_parent = {}
        for s in S:
            data_parent.update({s: np.ones(len(seq) - 1)})
        data_child = {}
        data_condition = {}
        for s in S:
            data_child.update({s: np.ones(len(seq) - 1)})
            data_condition.update({s: {}})
            if state is not None:
                state.reset_index(inplace=True, drop=True)
                for st in state.columns:
                    data_condition[s].update({st: np.ones(len(seq) - 1)})

        if state is not None:
            data_policy_condition = {}
            for s in state.columns:
                data_policy_condition.update({s: np.ones(len(seq) - 1)})
        else:
            data_condition = None
            data_policy_condition = None

        #
        for i in range(1, len(seq)):
            data_parent[seq[i - 1]][i - 1] = 2

            # If it appears within 1 to 6 steps after i, it is an occurrence and does not have to be adjacent.
            for j in range(i, min(i + 6, len(seq))):
                if data_child[seq[j]][i - 1] != 2:
                    data_child[seq[j]][i - 1] = 2
                    if state is not None:
                        for s in state.columns:
                            data_condition[seq[j]][s][i - 1] = state[s].iloc[j] + 1
            if state is not None:
                for s in state.columns:
                    data_policy_condition[s][i - 1] = state[s].iloc[i - 1] + 1

        data_parent = pd.DataFrame(data_parent, dtype=np.int64)
        data_child = pd.DataFrame(data_child, dtype=np.int64)
        if state is not None:
            data_condition = pd.DataFrame(data_condition, dtype=np.int64)
            data_policy_condition = pd.DataFrame(data_policy_condition, dtype=np.int64)
        condition_state = None
        if state is not None:
            #Get the state related to each strategy
            data = pd.concat([data_policy_condition, data_parent], axis=1).values.T
            data = np.array(data, dtype=np.int64)
            nstates = np.max(data, axis=1).T
            nstates = np.array(nstates, dtype=np.int64)
            casual_num = data_policy_condition.shape[1]
            effect_num = data_parent.shape[1]
            Alearn, _, _, _ = learnBayesNet_noparallelize(data=data, nstates=nstates,
                                                          maxNparents=[casual_num] * (casual_num + effect_num),
                                                          casual_num=casual_num, effect_num=effect_num)
            # print("condition_state completed!")
            condition_state = []
            names = np.array(list(data_condition.columns))
            for i in range(casual_num, casual_num + effect_num):
                index = np.where(Alearn[:, i] == 1)[0]
                index = list(names[index])
                condition_state.append(index)
        # print(condition_state)
        return data_child, data_parent, data_condition, condition_state

    def skip_gram(self, result, indexN, alpha=10):
        sequence = result["seq"]
        states = result["state"]
        S = result["S"]
        grammar = result["sets"]

        seqLen = [len(grammar[S.index(s)]) for s in sequence]

        num = np.sum(seqLen) + len(indexN)

        sum = -1
        pointer = 0
        newSeq = []
        for i in range(len(sequence)):
            sum += len(grammar[S.index(sequence[i])])
            newSeq.append(sequence[i])
            if pointer < len(indexN) and sum >= indexN[pointer]:
                newSeq.append("N")
                sum += 1
                pointer += 1

        N = np.array([1] * len(newSeq))
        EA = np.array([1] * len(newSeq))

        for i in range(len(newSeq)):
            if newSeq[i] != "N":
                continue
            N[i] = 2
            for j in range(i + 2, min(i + 6, len(newSeq))):
                if newSeq[j] != "N" and "EA" == grammar[S.index(newSeq[j])]:
                    EA[i] = 2
                    break

        EA = EA.reshape(-1, 1).T
        N = N.reshape(-1, 1).T
        EAStates = np.int64(np.max(EA).T)
        NStates = np.int64(np.max(N).T)

        U = alpha  # / (np.prod(EAStates))
        score2, _ = BDscore(EA.reshape(-1, 1), [], EAStates, [], U)

        U = alpha  # / (np.prod(EAStates) * np.prod(NStates))
        score1, U = BDscore(EA, N, EAStates, [NStates], U)
        if score2 / score1 > 1 and U[1, 1] / len(newSeq) > 0.025:
            # print("NEA", U[1, 1], len(newSeq), U[1, 1] / len(newSeq))
            return True, U[1, 1]
        return False, 0

    def Chunking(self, seq, S, state, condition, save_name, clusterFileNames, alpha=0.1, simulation=True,
                 real_set=None):
        if real_set is not None:
            sets, pro, position_gram, frequence, coverSet = self.parse_pro(seq, real_set, True)
            return sets, pro, coverSet
            x = 0
        gramLen = {s: 1 for s in S}
        sequence = copy.deepcopy(seq)  # Keep the original sequence, and divide it with the original sequence for each parse
        states = copy.deepcopy(state)
        P = self.tools.static_pro(seq, S)  # Get the original grammar-set S and the probability of each grammar appearing P
        I = 0
        num = 0  # Each combination of grammar is replaced by a serial number, starting from 0
        place_set = [chr(i) for i in range(32, 126)]
        place_set.remove('e')
        place_set.remove('G')
        place_set.remove("L")
        place_set.remove("E")
        place_set.remove("A")
        place_set.remove("1")
        place_set.remove("2")
        place_set.remove("3")
        place_set.remove("4")
        place_set.remove("S")
        place_set.remove("V")
        place_set.remove("N")
        place_set.remove(" ")
        place_set.remove("#")
        aggregate_dict = {}  # Record combination grammar
        cover_set, _ = self.get_cover_set(S, aggregate_dict, gramLen)
        predict_sets, predict_pro, _, _, _ = self.parse_pro(sequence, cover_set)
        # print(predict_sets)
        # print(predict_pro)
        P_ = {}
        for j, key in enumerate(predict_sets):
            P_.update({key: predict_pro[j]})
        kl_diff = []
        components = [[s, ''] for s in S]
        while I < 100000:
            component = []
            data_child, data_parent, data_condition, condition_state = self.organize_data(seq, S, state, condition)
            ratios = []
            chunks = []
            for i, cr in enumerate(S):
                if cr == "V" or cr == "1" or cr == "2" or cr == "N" or cr == "S" or cr == "e" or cr == "#":
                    continue
                data_r = data_child[cr].values
                nstates_r = np.int64(np.max(data_r).T)
                if data_condition is not None:
                    if len(condition_state[i]) != 0:
                        data_c = data_condition[condition_state[i]].values.T  # Child node data
                        nstates_c = np.array(np.max(data_c, 1).T, dtype=np.int64)
                    else:
                        data_c = []
                        nstates_c = []
                    if alpha < 0:
                        U = 1
                    else:
                        U = alpha  # / (np.prod(nstates_r) * (np.prod(nstates_c)))
                    score2, _ = BDscore(data_r, data_c, nstates_r, nstates_c, U)  # BDscore when the node has no parent node
                else:
                    if alpha < 0:
                        U = 1
                    else:
                        U = alpha  # / (np.prod(nstates_r))
                    score2, _ = BDscore(data_r, [], nstates_r, [], U)
                for j, cl in enumerate(S):
                    tempCr = self.deep(cr, aggregate_dict)
                    tempCl = self.deep(cl, aggregate_dict)
                    tempSet = list(set(list(tempCr)) & set(list(tempCl)))  #
                    if cl == cr or cl == "V" or len(tempSet) != 0 or cl == "N" or cl == "#":
                        continue
                    data_l = data_parent[cl].values.reshape(1, -1)  #
                    nstates_l = np.int64(np.max(data_l).T)
                    if data_condition is not None:
                        if len(condition_state[i]) != 0:
                            data_ = np.vstack((data_l, data_c))
                            data_ = np.array(data_, dtype=np.int64)
                            nstates_ = np.array(np.max(data_, 1).T, dtype=np.int64)
                        else:
                            data_ = np.array(data_l, dtype=np.int64)
                            nstates_ = nstates_l
                        if alpha < 0:
                            U = 1
                        else:
                            U = alpha  # / (np.prod(nstates_r) * (np.prod(nstates_)))
                        score1, U = BDscore(data_r, data_, nstates_r, nstates_, U)  #
                        _, U = BDscore(data_r, data_l, 2, 2, 1)
                    else:
                        if alpha < 0:
                            U = 1
                        else:
                            U = alpha  # / 4
                        score1, U = BDscore(data_r, data_l, 2, 2, U)

                    if U[1, 1] / len(seq) < P[i] * P[j] or U[1, 1] / len(seq) < 0.05:
                        continue

                    ratio = score2 / score1
                    ratios.append(ratio)
                    chunks.append(cl + cr)
                    component.append([cl, cr])
            if len(ratios) == 0:
                break
            chunks, ratios, component = self.tools.choice_max_n(ratios, chunks, component)
            if len(chunks) == 0:
                break
            flag = False

            for i, chunk in enumerate(chunks):
                if num > len(place_set) - 1:
                    flag = True
                    print("Not enough characters!" * 20)
                    break
                aggregate_dict.update({place_set[num]: chunk})
                S += [place_set[num]]
                components += [list(component[i])]
                gramLen.update({place_set[num]: max(gramLen[component[i][0]] + 1, gramLen[component[i][1]] + 1)})
                num += 1
            if flag == True:
                break
            # print(aggregate_dict)

            cover_set, _ = self.get_cover_set(S, aggregate_dict, gramLen)  #
            # print(cover_set)

            seq, state = self.parse(sequence, cover_set, S, states)  #

            P = self.tools.static_pro(seq, S)
            I += 1

            cover_set_, _ = self.get_cover_set(S, aggregate_dict, gramLen)  #
            Q_ = {}
            predict_sets, predict_pro, _, _, _ = self.parse_pro(sequence, cover_set_)
            for j, key in enumerate(predict_sets):
                if predict_pro[j] != 0:
                    Q_.update({key: predict_pro[j]})

            kl_value = self.tools.KL(Q_, P_)
            kl_diff.append(kl_value)
            P_ = copy.deepcopy(Q_)
            # print("kl_value", kl_value)
            if len(kl_diff) >= 5 and np.mean(kl_diff[-5:]) <= 0.05:
                print("Convergence!" * 20)
                break

        cover_set, Len = self.get_cover_set(S, aggregate_dict, gramLen)
        sets, pro, position_gram, frequence, coverSet = self.parse_pro(sequence, cover_set, True)
        components = [[self.deep(c[0], aggregate_dict), self.deep(c[1], aggregate_dict)] for c in components]

        deleteIndex = np.where(pro != 0)[0]
        S = np.array(S)[deleteIndex].tolist()
        sets = np.array(sets)[deleteIndex].tolist()
        pro = np.array(pro)[deleteIndex].tolist()
        frequence = np.array(frequence)[deleteIndex].tolist()
        frequency = deepcopy(frequence)
        components = np.array(components)[deleteIndex].tolist()

        for i in range(len(sets)):
            frequence[i] *= len(sets[i])
        time_pro = frequence / np.sum(frequence)
        result = {
            "sets": sets,
            "pro": pro,
            "gram": position_gram,
            "sequence": sequence,
            "time_pro": time_pro,
            "frequency": frequency,
            "seq": seq,
            "state": state,
            "S": S,
            "fileNames": clusterFileNames,
            "components": components
        }

        # print("time pro:", components)
        if simulation == True:
            return sets, pro, coverSet
        with open(save_name, "wb") as f:
            pickle.dump(result, f)
        return sets, pro, Len, components


class generator_recover:

    def __init__(self, sample_num=10000, processes_num=16, table_num=100):
        self.sample_num = sample_num
        self.processes_num = processes_num
        self.table_num = table_num

    def parallelize_recover(self, possibility, table, Type, need_condition):

        ID = possibility[0]

        possibility = possibility[1]
        sample_num = self.sample_num
        simulator = Simulator_grammar(None, Type)
        simulator.policy_table = table
        simulator.possibility = possibility

        re_generate = True
        while re_generate:
            if Type == "bi":
                seq, S, state, real_pro, actionList = simulator.generator_from_simulation_envoroment(2,
                                                                                                     sample_num=sample_num)
                sequence = copy.deepcopy(seq)
            elif Type == "tri":
                seq, S, state, real_pro, actionList = simulator.generator_from_simulation_envoroment(3,
                                                                                                     sample_num=sample_num)
                sequence = copy.deepcopy(seq)
            elif Type == "one":
                seq, S, state, real_pro, actionList = simulator.generator_from_simulation_envoroment(1,
                                                                                                     sample_num=sample_num)
                sequence = copy.deepcopy(seq)
            re_generate = False
        chunk = Chunk()
        real_set = list(real_pro.keys())
        if need_condition == True:
            if state is not None:
                data = state.values.T + 1
                # idx = np.where(np.array(list(seq)) != "#")[0]
                # data = data[:, idx]
                G = PC(data, data.shape[1])
                # G = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                condition = []
                for i in range(len(G)):
                    nondes = list(np.where(G[i, :] == 1)[0])
                    condition.append(nondes)
            else:
                condition = None
            sets, predict_pro, coverSet = chunk.Chunking(seq, S, state, condition, None, None, alpha=0.5,
                                                         simulation=True)
        else:
            sets, predict_pro = chunk.Chunking(seq, S, state=None)

        print("ID", ID, sets, np.mean([len(c) for c in coverSet]), np.sum(G))
        return sequence, real_pro, sets, predict_pro, actionList, coverSet

    def generate_possibility(self, possible_choice, possible_assign, sample_num=100):
        """
        Based on the possible strategies of each state and their restrictions, the probability of each strategy under each state is generated.
        :param possible_choice:
        :param possible_assign:
        :return:
        """

        def need_refuse(states, pro):
            pro_ = {"G": 0, "L": 0, "e": 0, "E": 0, "A": 0}
            for i, state in enumerate(states):
                strategies = possible_choice[state]
                for j, strategy in enumerate(strategies):
                    for s in strategy:
                        pro_[s] += pro[i][j]
            pro_ = np.array(list(pro_.values()))
            pro_ = pro_ / np.sum(pro_)
            std = np.std(pro_)
            if std < 0.1:
                return False, std
            else:
                return True, std

        def similarity(possibilities, pro, possible_assign):
            possibilities = np.array(possibilities)
            possible_assign = np.array(list(possible_assign.values()))
            pro = np.array(pro)
            flag = False
            for i in range(possibilities.shape[0]):
                p = np.abs(possibilities[i] - pro)
                s = 0
                for j in range(p.shape[0]):
                    temp1 = p[j][:int(possible_assign[j][1])]
                    temp2 = p[j][int(possible_assign[j][1]):]
                    index1 = len(np.where(temp1 < 0.05)[0])
                    index2 = len(np.where(temp2 < 0.01)[0])
                    if len(temp1) != 1:
                        s += index1
                    s += index2
                s /= (p.shape[0] * p.shape[1])
                if s > 0.3:
                    flag = True
                    break
            return flag

        states = list(possible_choice.keys())
        possibilities = []
        stds = []

        flag = True
        while flag:
            num = 0
            while num < 1000:
                pros = {}
                num += 1
                for i, state in enumerate(states):
                    possibility_prefer = possible_assign[state][0]
                    num_prefer = possible_assign[state][1]
                    pro_prefer = np.random.random(size=num_prefer)
                    pro_prefer = pro_prefer / np.sum(pro_prefer)
                    pro_prefer = pro_prefer * possibility_prefer

                    possibility_disgust = 1 - possibility_prefer
                    num_disgust = len(possible_choice[state]) - num_prefer

                    pro_disgust = np.random.random(size=num_disgust)
                    pro_disgust = pro_disgust / np.sum(pro_disgust)
                    pro_disgust = pro_disgust * possibility_disgust

                    pro = list(pro_prefer) + list(pro_disgust)
                    pros.update({state: deepcopy(pro)})
                refuse, std = need_refuse(states, list(pros.values()))
                possibilities_n = [list(p.values()) for p in possibilities]
                # similar = similarity(possibilities_n, list(pros.values()), possible_assign)
                if refuse == False:
                    stds.append(std)
                    possibilities.append(deepcopy(pros))
            # print(len(stds))
            if len(stds) > sample_num:
                flag = False
        index = sorted(enumerate(stds), key=lambda x: x[1])
        index = [i[0] for i in index][:sample_num]
        possibilities = list(np.array(possibilities)[index])
        return possibilities

    def get_choice_posibility_table(self, Type):
        states = [
            "0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111",
            "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111",
        ]
        if Type == "one":
            all_strategy = ["G", "L", "e", "E", "A"]
            possible_choice = {
                "0000": ["e", "E"], "0001": ["e"], "0010": ["A", "G", "E"], "0011": ["A", "G"],
                "0100": ["G", "E"], "0101": ["G"], "0110": ["G", "E"], "0111": ["G"],
                "1000": ["e", "E"], "1001": ["e"], "1010": ["A", "L", "E"], "1011": ["A", "L"],
                "1100": ["L", "E"], "1101": ["L"], "1110": ["L", "E"], "1111": ["L"]
            }
            possible_assign = {
                "0000": [0.8, 2], "0001": [0.8, 1], "0010": [0.8, 3], "0011": [0.8, 2],
                "0100": [0.8, 2], "0101": [0.8, 1], "0110": [0.8, 2], "0111": [0.8, 1],
                "1000": [0.8, 2], "1001": [0.8, 1], "1010": [0.8, 3], "1011": [0.8, 2],
                "1100": [0.8, 2], "1101": [0.8, 1], "1110": [0.8, 2], "1111": [0.8, 1]
            }
            savename = "../data/new_simulation/one_gram_possibility.pkl"
        elif Type == "bi":
            all_strategy = ["eG", "eE", "eL", "EA", "LG", "GL"]
            possible_choice = {
                "0000": ["eG", "eE", "EA"], "0001": ["eG"], "0010": ["EA", "GL"], "0011": ["GL"],
                "0100": ["GL", "EA"], "0101": ["GL"], "0110": ["GL", "EA"], "0111": ["GL"],
                "1000": ["EA", "eL", "eE"], "1001": ["eL"], "1010": ["EA", "LG"], "1011": ["LG"],
                "1100": ["LG", "EA"], "1101": ["LG"], "1110": ["LG", "EA"], "1111": ["LG"]
            }
            possible_assign = {
                "0000": [0.8, 3], "0001": [0.8, 1], "0010": [0.8, 2], "0011": [0.8, 1],
                "0100": [0.8, 2], "0101": [0.8, 1], "0110": [0.8, 2], "0111": [0.8, 1],
                "1000": [0.8, 3], "1001": [0.8, 1], "1010": [0.8, 2], "1011": [0.8, 1],
                "1100": [0.8, 2], "1101": [0.8, 1], "1110": [0.8, 2], "1111": [0.8, 1]
            }
            savename = "../data/new_simulation/bi_gram_possibility.pkl"
        elif Type == "tri":
            all_strategy = ["eGL", "eEA", "eLG", "EAG", "EAL", "ALG", "AGL"]
            # for stra in all_strategy:
            #     for strb in all_strategy:
            #         s = stra + strb
            #         s = s[1:4]
            #         if s in all_strategy:
            #             print(stra, strb, s)
            possible_choice = {
                "0000": ["eGL", "eEA"], "0001": ["eGL"], "0010": ["EAG", "AGL"], "0011": ["AGL"],
                "0100": ["eGL", "EAG"], "0101": ["eGL"], "0110": ["AGL", "EAG"], "0111": ["AGL"],
                "1000": ["EAL", "eLG", "eEA"], "1001": ["eLG"], "1010": ["EAL", "ALG"], "1011": ["ALG"],
                "1100": ["eLG", "EAL"], "1101": ["eLG"], "1110": ["ALG", "EAL"], "1111": ["ALG"]
            }
            possible_assign = {
                "0000": [0.8, 2], "0001": [0.8, 1], "0010": [0.8, 2], "0011": [0.8, 1],
                "0100": [0.8, 2], "0101": [0.8, 1], "0110": [0.8, 2], "0111": [0.8, 1],
                "1000": [0.8, 3], "1001": [0.8, 1], "1010": [0.8, 2], "1011": [0.8, 1],
                "1100": [0.8, 2], "1101": [0.8, 1], "1110": [0.8, 2], "1111": [0.8, 1]
            }
            savename = "../data/new_simulation/tri_gram_possibility.pkl"

        for state in states:
            temp1 = possible_choice[state]
            temp2 = deepcopy(all_strategy)
            for t in temp1:
                temp2.remove(t)
            possible_choice[state] = temp1 + temp2
        possibilities = self.generate_possibility(possible_choice, possible_assign, self.table_num)
        result = {
            "possible_assign": possible_assign,
            "possible_choice": possible_choice,
            "possibilities": possibilities
        }
        with open(savename, "wb") as file:
            pickle.dump(result, file)

    def recover(self, Type, need_condition=True):
        with open("../data/new_simulation/" + Type + "_gram_possibility.pkl", "rb") as file:
            result = pickle.load(file)
        possible_choice = result["possible_choice"]
        possibilities = result["possibilities"]
        possible_assign = result["possible_assign"]
        index = np.random.randint(0, len(possibilities), size=5000)
        possibilities = [possibilities[i] for i in index]

        with multiprocessing.Pool(processes=18) as pool:
            result = pool.map(
                partial(self.parallelize_recover, table=possible_choice, Type=Type, need_condition=need_condition),
                enumerate(possibilities))
        real_sequence = [r[0] for r in result]
        real_pro = [r[1] for r in result]
        predict_sets = [r[2] for r in result]
        predict_pro = [r[3] for r in result]
        actionList = [r[4] for r in result]
        coverSet = [r[5] for r in result]
        result = {
            "possible_choice": possible_choice,
            "possible_assign": possible_assign,
            "possibilities": possibilities,
            "real_sequence": real_sequence,
            "real_pro": real_pro,
            "predict_sets": predict_sets,
            "predict_pro": predict_pro,
            "actionList": actionList,
            "coverSet": coverSet
        }
        #######################################
        lops_c = [np.mean([len(c) for c in c_s]) for c_s in coverSet]
        data = pd.DataFrame({"lops_c": lops_c})
        data.to_csv("lops_c.csv")
        return result
        # with open(savename, "wb") as file:
        #     pickle.dump(result, file)


def getConditionGraph(fileName, dataType):
    if dataType == "human":
        with open("../data/statesPerCluster/" + fileName, "rb") as file:
            result = pickle.load(file)
    else:
        with open("../data/states monkey/monkey.pkl", "rb") as file:
            result = pickle.load(file)
    G = result["G"]
    condition = []
    for i in range(len(G)):
        nondes = list(np.where(G[i, :] == 1)[0])
        condition.append(nondes)

    return condition

if __name__ == '__main__':
    # test()
    generator_recover_ = generator_recover(processes_num=12, table_num=500, sample_num=1000)
    generator_recover_.get_choice_posibility_table(Type="one")
    generator_recover_.get_choice_posibility_table(Type="bi")
    generator_recover_.get_choice_posibility_table(Type="tri")

    results = {}
    num_samples = list(range(100, 1001, 100))
    # num_samples = list(range(1000, 3001, 1000))
    for num_sample in num_samples:
        print(num_sample, "=====" * 100)
        generator_recover_ = generator_recover(processes_num=12, table_num=500, sample_num=num_sample)
        result = generator_recover_.recover(Type="tri", need_condition=True)
        results.update({num_sample: result})
    with open("../data/simulation/simResult-tri.pkl", "wb") as file:
        pickle.dump(results, file)

    for num_sample in num_samples:
        print(num_sample, "=====" * 100)
        generator_recover_ = generator_recover(processes_num=12, table_num=500, sample_num=num_sample)
        result = generator_recover_.recover(Type="bi", need_condition=True)
        results.update({num_sample: result})
    with open("../data/simulation/simResult-bi.pkl", "wb") as file:
        pickle.dump(results, file)
    #
    for num_sample in num_samples:
        print(num_sample, "=====" * 100)
        generator_recover_ = generator_recover(processes_num=12, table_num=500, sample_num=num_sample)
        result = generator_recover_.recover(Type="one", need_condition=True)
        results.update({num_sample: result})
    with open("../data/simulation/simResult-one.pkl", "wb") as file:
        pickle.dump(results, file)
    # #
