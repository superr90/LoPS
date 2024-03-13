import os
import pandas as np
import numpy as np
import copy
from copy import deepcopy
import pandas as pd
from functools import partial
import multiprocessing
import pickle
import zlib


def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if x >= lower_bound and x <= upper_bound]


def remove_outliers_using_zscore(data, threshold=2):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    return [x for x, z in zip(data, z_scores) if -threshold <= z <= threshold]


def kolmogorov_complexity(data):
    data = "".join([str(int(d)) for d in data])
    compressed_data = zlib.compress(data.encode())
    return len(compressed_data)


final_cover = []
probility = []


def skipGramLen(cover, coverLen):
    for i, s in enumerate(cover):
        if s == "N":
            pre = min(i + 2, len(cover) - 1)
            tail = min(i + 6, len(cover) - 1)
            tempCover = cover[pre:tail]
            nInCover = ["N" in t for t in tempCover]
            if True in nInCover:
                index = nInCover.index(True)
                tempCover = tempCover[:index]
            for tempc in tempCover:
                if "EA" in tempc:
                    EAindex = tempCover.index(tempc) + 2
                    coverLen[i] = EAindex
                    break
    return coverLen


def cover(sets, strategy, P, needSipGram=False):
    global final_cover, probility
    final_cover = []
    probility = []

    # Search deeply to find the cover set with the highest probability
    def deep(seq, current_cover, P):
        if len(seq) == 0:
            global probility, final_cover
            pro = 1
            for c in current_cover:
                index = sets.index(c)
                pro *= P[index]
            final_cover.append(deepcopy(current_cover))
            probility.append(pro)
            return
        for j, s in enumerate(sets):
            l = len(s)
            if len(seq) >= l and seq[:l] == s:
                current_cover.append(s)
                deep(deepcopy(seq[l:]), deepcopy(current_cover), deepcopy(P))
                current_cover = current_cover[:-1]

    # strategy sequence conversion
    Dict = {
        0: "G", 1: "L", 2: "1", 3: "2", 4: "3", 5: "4", 6: "A", 7: "E", 8: "N", 9: "V", 10: "S"
    }
    seqs = [Dict[s] for s in strategy]

    # strategy sequence deduplication
    seq = [seqs[0]]
    index = [0]
    pre = seqs[0]
    for i in range(1, len(seqs)):
        if seqs[i] == pre:
            continue
        seq.append(seqs[i])
        index.append(i)
        pre = seqs[i]
    seq = "".join(seq)
    deep(seq, [], P)
    index_ = np.argmax(probility)
    result_cover = final_cover[index_]

    # Divide new breakpoints based on coverage clearing (grammar starting position)
    new_index = []
    pointer = 0
    for c in result_cover:
        new_index.append(index[pointer])
        pointer += len(c)

    # Assign grammar to each step
    coverLen = [len(r) for r in result_cover]
    # coverLen = [3 if (l == 3 or l == 4) else l for l in coverLen]
    if needSipGram == True:
        # If it is N, determine whether it is skip-gram.
        coverLen = skipGramLen(result_cover, coverLen)
    gram = []
    gram_num = []
    gramStart = []
    gramLen = []
    for i in range(len(seqs)):
        if i in new_index:
            t = new_index.index(i)
            gram.append(result_cover[t])
            gram_num.append(0)
            gramStart.append(1)
            gramLen.append(coverLen[t])
        else:
            gramStart.append(0)
            gram.append(gram[-1])
            gramLen.append(gramLen[-1])
            if seqs[i] != seqs[i - 1]:
                gram_num.append(gram_num[-1] + 1)
            else:
                gram_num.append(gram_num[-1])

    return gram, gram_num, gramStart, gramLen


def Parsing(filePath, sets, saveFolder, needSipGram):
    print(filePath)
    if "N" not in sets:
        sets.append("N")
    data = pd.read_pickle(filePath)
    data["game"] = data.file.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )
    Dict = {
        0: "G", 1: "L", 2: "1", 3: "2", 4: "3", 5: "4", 6: "A", 7: "E", 8: "N", 9: "V", 10: "S"
    }
    newData = []
    for idx, grp in data.groupby("game"):
        if grp["ifscared3"].iloc[0] != -1:
            continue
        strategy = grp["strategy"]
        strategySeq = [Dict[s] for s in strategy]

        # strategySeq removes duplicate content and marks the position
        seq = [strategySeq[0]]
        index = [0]
        pre = strategySeq[0]
        for i in range(1, len(strategySeq)):
            if strategySeq[i] == pre:
                continue
            seq.append(strategySeq[i])
            index.append(i)
            pre = strategySeq[i]
        seq = "".join(seq)

        # maximum coverage
        resultCover = []
        i = 0
        while i < len(seq):
            maxLenSet = ""
            for s in sets:
                if len(s) > len(maxLenSet) and len(s) <= len(seq) - i and seq[i:i + len(s)] == s:
                    maxLenSet = s
            i += len(maxLenSet)
            resultCover.append(maxLenSet)

        # Set grammar for each step based on the parsing results.
        new_index = []
        pointer = 0
        for c in resultCover:
            new_index.append(index[pointer])
            pointer += len(c)

        # Assign grammar to each step
        coverLen = [len(r) for r in resultCover]
        # If it is N, determine whether it is skip-gram.
        if needSipGram == True:
            # If it is N, determine whether it is skip-gram.
            coverLen = skipGramLen(resultCover, coverLen)
        gram = []
        gram_num = []
        gramStart = []
        gramLen = []
        for i in range(len(strategySeq)):
            if i in new_index:
                t = new_index.index(i)
                gram.append(resultCover[t])
                gram_num.append(0)
                gramStart.append(1)
                gramLen.append(coverLen[t])
            else:
                gramStart.append(0)
                gram.append(gram[-1])
                gramLen.append(gramLen[-1])
                if strategySeq[i] != strategySeq[i - 1]:
                    gram_num.append(gram_num[-1] + 1)
                else:
                    gram_num.append(gram_num[-1])
        grp["gram"] = gram
        grp["gram_num"] = gram_num
        grp["gramStart"] = gramStart
        grp["gramLen"] = gramLen
        newData.append(copy.deepcopy(grp))
    newData = pd.concat(newData)
    newData.reset_index(inplace=True, drop=True)

    savePath = saveFolder + "-".join(filePath.split("/")[-1].split("-")[:2]) + ".pkl"
    newData.to_pickle(savePath)


def ParsingMain():
    grammarPath = "../HumanData/GrammarFinall/"
    fileFolder = "../HumanData/CorrectedWeightData/"
    saveFolder = "../HumanData/GrammarData/"

    fileNames = os.listdir(fileFolder)
    grammarFileNames = os.listdir(grammarPath)
    fileNames.sort()
    grammarFileNames.sort()
    for k in range(len(fileNames)):
        result = pd.read_pickle(grammarPath + grammarFileNames[k])
        filePath = fileFolder + fileNames[k]
        Parsing(filePath, sets=result["sets"], saveFolder=saveFolder, needSipGram=result["skipGram"])


def LoPSComplexity(filePath, sets, skipGram):
    """
    Average complexity per game
    """
    print(filePath)
    if "N" not in sets:
        sets.append("N")
    data = pd.read_pickle(filePath)
    data["game"] = data.file.str.split("-").apply(
        lambda x: "-".join([x[0]] + x[2:])
    )

    Dict = {
        0: "G", 1: "L", 2: "1", 3: "2", 4: "3", 5: "4", 6: "A", 7: "E", 8: "N", 9: "V", 10: "S"
    }

    allDepth = []
    meanDepthPerGame = []
    for idx, grp in data.groupby("game"):
        if grp["ifscared3"].iloc[0] != -1:
            continue
        stratgy = list(grp["strategy"])
        stratgy = [Dict[s] for s in stratgy]
        stratgy = [s for i, s in enumerate(stratgy) if i == 0 or s != stratgy[i - 1]]
        seq = "".join(stratgy)
        indexN = np.where(np.array(list(seq)) == "N")[0]

        seq = seq.replace("N", "")

        resultCover = []
        i = 0
        while i < len(seq):
            maxLenSet = ""
            for s in sets:
                if len(s) > len(maxLenSet) and len(s) <= len(seq) - i and seq[i:i + len(s)] == s:
                    maxLenSet = s
            i += len(maxLenSet)
            resultCover.append(maxLenSet)

        coverLen = [len(c) for c in resultCover]
        coverLen_ = [len(c) for c in resultCover if c != "V"]
        coverDepth = [cl for cl in coverLen_]
        if skipGram == True:
            newSeq = []
            pointer = 0
            sum = -1
            for i in range(len(coverLen)):
                sum += len(resultCover[i])
                newSeq.append(resultCover[i])
                if pointer < len(indexN) and sum >= indexN[pointer]:
                    newSeq.append("N")
                    sum += 1
                    pointer += 1
            skipGramLen = []
            for i in range(len(newSeq)):
                if newSeq[i] == "N":
                    skipGramLen.append(1)
                    for j in range(i + 2, min(i + 6, len(newSeq))):
                        if "N" in newSeq[j]:
                            break
                        if "EA" in newSeq[j]:
                            skipGramLen[-1] = j - i + 1
                            break
        else:
            skipGramLen = [1] * len(indexN)
        skipGramDepth = [sl for sl in skipGramLen]
        depth = coverDepth + skipGramDepth
        meanDepth = np.mean(depth)
        allDepth += depth
        meanDepthPerGame.append(meanDepth)

    return meanDepthPerGame, allDepth


def LoPS_complexity_human(date):
    grammarPath = "../HumanData/GrammarFinall/" + date + "/"
    fileFolder = "../HumanData/CorrectedWeightData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    grammarFileNames = os.listdir(grammarPath)
    fileNames.sort()
    grammarFileNames.sort()
    for k in range(len(fileNames)):
        if k != 3:
            continue
        result = pd.read_pickle(grammarPath + grammarFileNames[k])
        filePath = fileFolder + fileNames[k]
        meanDepthPerGame, allDepth = LoPSComplexity(filePath, result["sets"], result["skipGram"])
        meanDepthPerSub = np.mean(allDepth)
        data = {
            "allDepth": allDepth,
            "meanDepthPerSub": meanDepthPerSub,
            "meanDepthPerGame": meanDepthPerGame,
            "fileNames": grammarFileNames[k]
        }
        path = "../HumanData/LoPSComplexity/" + grammarFileNames[k]
        pd.to_pickle(data, path)


if __name__ == '__main__':
    date = "session2"
    LoPS_complexity_human()
    # ParsingMain()
