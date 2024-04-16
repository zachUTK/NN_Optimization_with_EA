import pandas as pd
import os
import torch
import numpy as np
import math




def loadData(split):

    
    path = "./data/"
    csvFiles = [file for file in os.listdir(path) if file.endswith('.csv')]
    csvFiles.sort()
    trainX = pd.DataFrame()
    testX = pd.DataFrame()

    trainSplit = math.floor(len(csvFiles) * split)

    
    for i in range(len(csvFiles)):
        file = csvFiles[i]

        if(i <= trainSplit-1):
            pathToFile = os.path.join(path, file)
            csv = pd.read_csv(pathToFile)
            trainX = pd.concat([trainX, csv])
        else:
            pathToFile = os.path.join(path, file)
            csv = pd.read_csv(pathToFile)
            testX = pd.concat([testX, csv])


    trainX.reset_index(drop=True, inplace=True)
    trainX = trainX.drop(trainX.columns[0], axis=1)
    testX.reset_index(drop=True, inplace=True)
    testX = testX.drop(testX.columns[0], axis=1)

    trainY = trainX.iloc[:, 3:4]
    trainX = trainX.drop(trainX.columns[3], axis=1)
    normTrainX = normalizeData(trainX)

    testY = testX.iloc[:, 3:4]
    testX = testX.drop(testX.columns[3], axis=1)
    normTestX = normalizeData(testX)

    normTrainX = torch.tensor(normTrainX.values).float()
    normTestX = torch.tensor(normTestX.values).float()

    trainY = torch.tensor(trainY.values).float()
    testY = torch.tensor(testY.values).float()

    

    
    return normTrainX, trainY, normTestX, testY

def normalizeData(x):

    min_val = np.min(x)
    max_val = np.max(x)
    norm = (x - min_val) / (max_val - min_val)

    
    '''mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / std'''

    return norm



