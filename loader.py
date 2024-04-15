import pandas as pd
import os
import torch
import numpy as np




def loadData():

    path = "./data/"
    csvFiles = [file for file in os.listdir(path) if file.endswith('.csv')]
    csvFiles.sort()
    data = pd.DataFrame()

    
    for file in csvFiles:
        pathToFile = os.path.join(path, file)
        csv = pd.read_csv(pathToFile)
        data = pd.concat([data, csv])

    data.reset_index(drop=True, inplace=True)
    data = data.drop(data.columns[0], axis=1)

    Y = data.iloc[:, 3:4]
    X = data.drop(data.columns[3], axis=1)
    normX = normalizeData(X)



    

    #X = torch.tensor(X.values).float()
    #Y = torch.tensor(Y.values).float()


    return normX, Y

def normalizeData(x):

    min_val = np.min(x)
    max_val = np.max(x)
    norm = (x - min_val) / (max_val - min_val)

    
    '''mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / std'''

    return norm



