import pandas as pd
import torch


def loadData():
    data = pd.read_csv("./data/AAPL.csv")

    data = data.drop(data.columns[0], axis=1)

    trainY = data.iloc[:, 3:4]
    trainX = data.drop(data.columns[3], axis=1)

    trainX = torch.tensor(trainX.values).float()
    trainY = torch.tensor(trainY.values).float()


    return trainX, trainY

def normalizeData(data):

    min_val = torch.min(data)
    max_val = torch.max(data)
    data = (data - min_val) / (max_val - min_val)

    
    mean = torch.mean(data)
    std = torch.std(data)
    data = (data - mean) / std

    return data



