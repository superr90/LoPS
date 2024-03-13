import copy
import os
import pickle
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")


def CorrectPosition(data, frameData):
    """
    1.If the pacman trajectory is discontinuous, locate the original frame data position and extract the missing coordinates to ensure that the pacman trajectory is continuous.
    2.Correct the coordinates of the ghost that recorded errors. (14, 20), (15, 20), (16, 20) The second coordinate of these coordinates is modified to 19
    """
    keys = ["ghost1Pos", "ghost2Pos", "ghost3Pos", "ghost4Pos"]
    positionTypes = ["pacmanPos"]  # Data column names that need to be made continuous
    newSeriesList = []
    for i in range(len(data)):
        # Correct the incorrectly recorded ghost coordinates
        for key in keys:
            if data[key].iloc[i] in [(14, 20), (15, 20), (16, 20)]:
                data[key][i] = (data[key][i][0], 19)
        # If the pacman trajectory is discontinuous, locate the original frame data position and extract the missing coordinates to ensure that the pacman trajectory is continuous.
        if i == 0 or i == len(data) - 1:
            continue
        for _, positionType in enumerate(positionTypes):
            prePosition = data[positionType].iloc[i - 1]
            curPosition = data[positionType].iloc[i]
            if isinstance(prePosition, str):
                prePosition = eval(prePosition)
                curPosition = eval(curPosition)
            # Determine whether it is continuous
            if len(curPosition) == 0:
                continue
            if np.abs(prePosition[0] - curPosition[0]) + np.abs(prePosition[1] - curPosition[1]) == 1:
                continue
            if prePosition == (0, 18) and curPosition == (30, 18):
                continue
            if prePosition == (30, 18) and curPosition == (0, 18):
                continue
            # If not continuous, insert
            newSeries = None  #
            insertedPosition = []
            # Locate the position of the previous position and current position data in the frame data
            prePositionIndexInFrame = data["frameIndex"].iloc[i - 1]
            curPositionIndexInFrame = data["frameIndex"].iloc[i]
            # Get the corresponding data interval in frame data
            allFrameDataIndex = list(frameData.index)
            index1 = allFrameDataIndex.index(prePositionIndexInFrame)
            index2 = allFrameDataIndex.index(curPositionIndexInFrame)
            frameIndex = allFrameDataIndex[index1:index2]
            # Search within the corresponding frame data data interval to find a coordinate that makes the pacman trajectory continuous.
            for j in frameIndex:
                tempPosition = frameData[positionType].loc[j]
                if isinstance(tempPosition, str):
                    tempPosition = eval(tempPosition)
                # 若
                if tempPosition != prePosition and tempPosition != curPosition and tempPosition != (
                        -1, 18) and tempPosition != (31, 18) and tempPosition not in insertedPosition:
                    newSeries = copy.deepcopy(frameData.loc[j])
                    insertedPosition.append(copy.deepcopy(tempPosition))
            if newSeries is not None:
                if len(insertedPosition) > 1:
                    print("=================" * 10)
                newSeriesList.append([newSeries, i + data.index[0], positionType])

    df = copy.deepcopy(data)
    for i in range(len(newSeriesList)):
        newSeries = newSeriesList[i][0]
        insertPosition = newSeriesList[i][1]
        newSeries["frameIndex"] = deepcopy(newSeries['Unnamed: 0'])
        df = df.loc[:insertPosition - 1].append(newSeries).append(df.loc[insertPosition:])
    return df


def getDir(pos1, pos2):
    """
    Get the direction of motion based on two adjacent positions
    """
    if pos1 == (0, 18) and pos2 == (30, 18):
        offset = (-1, 0)
    elif pos1 == (30, 18) and pos2 == (0, 18):
        offset = (1, 0)
    else:
        offsetX = pos2[0] - pos1[0]
        offsetY = pos2[1] - pos1[1]
        offset = (offsetX, offsetY)

    Dict = {
        (-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down", (0, 0): np.nan
    }

    return Dict[offset]


def CorrectDir(data):
    pacmanPos = list(data["pacmanPos"])
    pacmanPos = [eval(g) for g in pacmanPos if isinstance(g, str)]
    dir = [np.nan]
    for i in range(1, len(data)):
        dir.append(getDir(pacmanPos[i - 1], pacmanPos[i]))

    data["pacman_dir"] = dir
    return data


def CorrectTileData(date):
    """
    Correct the data:
     1. For the case where the pacman trajectory is discontinuous, locate the original frame data position and extract the missing coordinates to ensure that the pacman trajectory is continuous.
     2. Redetermine the movement direction of pacman based on the position of pacman
     3. Correct the left side of the ghost that recorded the error. The second coordinates of (14, 20), (15, 20), (16, 20) are all changed to 19.
    """

    # path="./fmriFrameData/"
    fileFolder = "../HumanData/TileData/" + date + "/"
    pixFilefolder = "../HumanData/FrameData/" + date + "/"
    fileNames = os.listdir(fileFolder)
    for i, fileName in enumerate(fileNames):
        print(fileName)
        # read tile data
        with open(fileFolder + fileName, "rb") as file:
            data = pickle.load(file)
        data.reset_index(drop=True, inplace=True)
        # frameIndex is the position of tile data in frame data
        data["frameIndex"] = data['Unnamed: 0']

        # read frame data
        with open(pixFilefolder + fileName, "rb") as file:
            frameData = pickle.load(file)
        frameData.reset_index(drop=True, inplace=True)

        correctedData = []
        for idx, grp in data.groupby("DayTrial"):
            frameGrp = frameData[frameData.DayTrial == idx]
            group = CorrectPosition(grp, frameGrp)
            group = CorrectDir(group)
            correctedData.append(group)
        if len(correctedData) != 0:
            data = pd.concat(correctedData)
            data.reset_index(drop=True, inplace=True)
            path = "../HumanData/CorrectedTileData/" + fileName
            data.to_pickle(path)


def ExtractTileFromFrame(date):
    """
   Extract frames from the original data, and extract the first frame every 25 frames as a tile data point.
    """
    sample_rate = 25
    filefolder = "../HumanData/FrameData/" + date + "/"
    savefolder = "../HumanData/TileData/" + date + "/"
    filenames = os.listdir(filefolder)

    for j, filename in enumerate(filenames):
        print(filename)
        data = pd.read_pickle(filefolder + filename)
        # Draw frames in round units
        new_data = []
        for i, grp in data.groupby("DayTrial"):

            idx = np.arange(0, grp.shape[0], sample_rate)
            if idx[-1] != grp.shape[0] - 1:
                idx = np.append(idx, grp.shape[0] - 1)
            grp = grp.iloc[idx]
            grp.reset_index(drop=True, inplace=True)
            new_data.append(deepcopy(grp))
        new_data = pd.concat(new_data, axis=0)
        new_data.reset_index(drop=True, inplace=True)

        # Delete the extracted points with coordinates (-1, 18) and (31, 18)
        pacmanPos = new_data["pacmanPos"].apply(lambda x: eval(x))
        index1 = np.where(pacmanPos == (-1, 18))[0]
        index2 = np.where(pacmanPos == (31, 18))[0]
        index = list(set(list(index1) + list(index2)))
        index.sort()
        new_data = new_data.drop(index)
        new_data.reset_index(drop=True, inplace=True)

        new_data.to_pickle(savefolder + filename)


def human_data_preprocess(date):
    ExtractTileFromFrame(date)
    CorrectTileData(date)


if __name__ == '__main__':
    date = "session2"
    human_data_preprocess(date)
