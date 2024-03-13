import copy
import os
import pickle
import scipy
import numpy as np
import re
import pandas as pd
import math
from PGM.bayesianScore import BDscore, learnBayesNet_noparallelize, learnBayesNetBlock
import random
import warnings
from functools import partial
import multiprocessing
import sys
from copy import deepcopy

sys.path.append("../")
warnings.filterwarnings("ignore")


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
        return sets, pro, position_gram, frequence

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

        data_parent = pd.DataFrame(data_parent, dtype=np.int)
        data_child = pd.DataFrame(data_child, dtype=np.int)
        if state is not None:
            data_condition = pd.DataFrame(data_condition, dtype=np.int)
            data_policy_condition = pd.DataFrame(data_policy_condition, dtype=np.int)
        condition_state = None
        if state is not None:
            # Get the state related to each strategy
            data = pd.concat([data_policy_condition, data_parent], axis=1).values.T
            data = np.array(data, dtype=np.int)


            nstates = np.max(data, axis=1).T
            nstates = np.array(nstates, dtype=np.int)
            casual_num = data_policy_condition.shape[1]
            effect_num = data_parent.shape[1]
            blockMessage = {i: [i] for i in range(casual_num)}

            Alearn, parameters, bestparents, bestscores = learnBayesNetBlock(data=data, nstates=nstates,
                                                                             blockMessage=blockMessage,
                                                                             casualNum=casual_num,
                                                                             blockNum=len(blockMessage),
                                                                             effectNum=effect_num, conditions=condition)

            condition_state = []
            names = np.array(list(data_condition.columns))
            for i in range(casual_num, casual_num + effect_num):
                index = np.where(Alearn[:, i] == 1)[0]
                index = list(names[index])
                condition_state.append(index)
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

            #If it appears within 1 to 6 steps after i, it is an occurrence and does not have to be adjacent.
            for j in range(i, min(i + 6, len(seq))):
                if data_child[seq[j]][i - 1] != 2:
                    data_child[seq[j]][i - 1] = 2
                    if state is not None:
                        for s in state.columns:
                            data_condition[seq[j]][s][i - 1] = state[s].iloc[j] + 1
            if state is not None:
                for s in state.columns:
                    data_policy_condition[s][i - 1] = state[s].iloc[i - 1] + 1

        data_parent = pd.DataFrame(data_parent, dtype=np.int)
        data_child = pd.DataFrame(data_child, dtype=np.int)
        if state is not None:
            data_condition = pd.DataFrame(data_condition, dtype=np.int)
            data_policy_condition = pd.DataFrame(data_policy_condition, dtype=np.int)
        condition_state = None
        if state is not None:
            # Get the state related to each strategy
            data = pd.concat([data_policy_condition, data_parent], axis=1).values.T
            data = np.array(data, dtype=np.int)
            nstates = np.max(data, axis=1).T
            nstates = np.array(nstates, dtype=np.int)
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
        EAStates = np.int(np.max(EA).T)
        NStates = np.int(np.max(N).T)

        U = alpha
        score2, _ = BDscore(EA.reshape(-1, 1), [], EAStates, [], U)

        U = alpha
        score1, U = BDscore(EA, N, EAStates, [NStates], U)
        if score2 / score1 > 1 and U[1, 1] / len(newSeq) > 0.025:
            return True, U[1, 1]
        return False, 0

    def Chunking(self, seq, S, state, condition, save_name, clusterFileNames, alpha=0.1):
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

        aggregate_dict = {}  # Record combination grammar
        cover_set, _ = self.get_cover_set(S, aggregate_dict, gramLen)
        predict_sets, predict_pro, _, _ = self.parse_pro(sequence, cover_set)
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
                if cr == "V" or cr == "1" or cr == "2" or cr == "N" or cr == "S" or cr == "e":
                    continue
                data_r = data_child[cr].values
                nstates_r = np.int(np.max(data_r).T)
                if data_condition is not None:
                    if len(condition_state[i]) != 0:
                        data_c = data_condition[condition_state[i]].values.T  # 子节点数据
                        nstates_c = np.array(np.max(data_c, 1).T, dtype=np.int)
                    else:
                        data_c = []
                        nstates_c = []
                    if alpha < 0:
                        U = 1
                    else:
                        U = alpha
                    score2, _ = BDscore(data_r, data_c, nstates_r, nstates_c, U)  # BDscore when the child node has no parent node
                else:
                    if alpha < 0:
                        U = 1
                    else:
                        U = alpha
                    score2, _ = BDscore(data_r, [], nstates_r, [], U)
                for j, cl in enumerate(S):
                    tempCr = self.deep(cr, aggregate_dict)
                    tempCl = self.deep(cl, aggregate_dict)
                    tempSet = list(set(list(tempCr)) & set(list(tempCl)))  # If there is an intersection, skip it过
                    if cl == cr or cl == "V" or len(tempSet) != 0 or cl == "N":
                        continue
                    data_l = data_parent[cl].values.reshape(1, -1)  # Parent node data
                    nstates_l = np.int(np.max(data_l).T)
                    if data_condition is not None:
                        if len(condition_state[i]) != 0:
                            data_ = np.vstack((data_l, data_c))
                            data_ = np.array(data_, dtype=np.int)
                            nstates_ = np.array(np.max(data_, 1).T, dtype=np.int)
                        else:
                            data_ = np.array(data_l, dtype=np.int)
                            nstates_ = nstates_l
                        if alpha < 0:
                            U = 1
                        else:
                            U = alpha  # / (np.prod(nstates_r) * (np.prod(nstates_)))
                        score1, U = BDscore(data_r, data_, nstates_r, nstates_, U)  # BDscore when the parent node is cl and the child node is cr
                        _, U = BDscore(data_r, data_l, 2, 2, 1)
                    else:
                        if alpha < 0:
                            U = 1
                        else:
                            U = alpha  # / 4
                        score1, U = BDscore(data_r, data_l, 2, 2, U)
                    if U[1, 1] / len(seq) < P[i] * P[j] or U[1, 1] / len(seq) < 0.05:  # If the number of occurrences of cl-->cr does not meet the requirements
                        continue
                    # When the child node has no parent node
                    ratio = score2 / score1
                    ratios.append(ratio)
                    chunks.append(cl + cr)
                    component.append([cl, cr])
            if len(ratios) == 0:
                break
            chunks, ratios, component = self.tools.choice_max_n(ratios, chunks, component)  # Select the largest n chunks
            if len(chunks) == 0:
                break
            flag = False
            # Update the selected chunks to aggregate_dict
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

            cover_set, _ = self.get_cover_set(S, aggregate_dict, gramLen)  # 获取当前的grammar集合
            # print(cover_set)

            seq, state = self.parse(sequence, cover_set, S, states)  # 对原始数据用新的grammar集合进行划分
            # Recalculate grammar
            P = self.tools.static_pro(seq, S)
            I += 1

            cover_set_, _ = self.get_cover_set(S, aggregate_dict, gramLen)  # 获取当前的grammar集合
            Q_ = {}
            predict_sets, predict_pro, _, _ = self.parse_pro(sequence, cover_set_)
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
        sets, pro, position_gram, frequence = self.parse_pro(sequence, cover_set, True)
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

        with open(save_name, "wb") as f:
            pickle.dump(result, f)
        return sets, pro, Len, components


def getConditionGraph(filePath):
    result = pd.read_pickle(filePath)
    G = result["G"]
    condition = []
    for i in range(len(G)):
        nondes = list(np.where(G[i, :] == 1)[0])
        condition.append(nondes)
    return condition


def grammar_induction_human(date):
    chunk = Chunk()
    fileFolder = "../HumanData/seq/" + date + "/"
    fileNames = os.listdir(fileFolder)
    stateFolder = "../HumanData/state/" + date + "/"
    saveFolder = "../HumanData/grammar/" + date + "/"
    stateNames = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]

    for i, fileName in enumerate(fileNames):
        result = pd.read_pickle(fileFolder + fileName)
        seq = result["seq"]
        S = result["S"]
        states = result["state"][stateNames]
        states.reset_index(inplace=True, drop=True)
        indexN = np.where((np.array(list(seq)) == "N"))[0]
        seq = seq.replace("N", '')
        states = states.drop(indexN)

        clusterFileNames = result["fileNames"]
        condition = getConditionGraph(stateFolder + fileName)
        sets, pro, Len, components = chunk.Chunking(seq, S, state=states, condition=condition,
                                                    save_name=saveFolder + fileName,
                                                    clusterFileNames=clusterFileNames, alpha=0.5)
        sets = sorted(sets)
        sets = sorted(sets, key=lambda x: len(x))
        print(sets)
        with open(saveFolder + fileName, "rb") as f:
            result = pickle.load(f)

        NEA, num = chunk.skip_gram(result, indexN, 0.5)
        if NEA == True:
            result["skipGram"] = True
        else:
            result["skipGram"] = False
        result["skipGramNum"] = num
        with open(saveFolder + fileName, "wb") as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    date = "session2"
    grammar_induction_human(date)
