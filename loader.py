import pandas as pd
import os
import numpy as np
import math

def load_data(split=0.8, data_path="./data/"):
    csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
    split_idx = math.floor(len(csv_files) * split)

    # Load and combine training/test files
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]

    train_df = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in train_files])
    test_df = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in test_files])

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Drop unwanted columns 
    cols_to_drop = [0, 4]  
    train_df.drop(train_df.columns[cols_to_drop], axis=1, inplace=True)
    test_df.drop(test_df.columns[cols_to_drop], axis=1, inplace=True)

    # Extract labels 
    target_col = 3
    trainY = train_df.iloc[:, target_col].values
    testY = test_df.iloc[:, target_col].values

    trainX = train_df.drop(train_df.columns[target_col], axis=1)
    testX = test_df.drop(test_df.columns[target_col], axis=1)

    # Normalize
    norm_trainX, mean, std = normalize(trainX)
    norm_testX = (testX - mean) / (std + 1e-8)

    return norm_trainX.values, trainY, norm_testX.values, testY


def normalize(df):
    mean = df.mean()
    std = df.std()
    norm_df = (df - mean) / (std + 1e-8)
    return norm_df, mean, std
