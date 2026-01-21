import os
import numpy as np
import pandas as pd
from config import config

def read_data(filename):
    return pd.read_csv(os.path.join(config.data_path, filename))

def get_data():
    data_list = []
    for filename in config.DATA:
        df = read_data(filename)
        data_list.append(df)
    return pd.concat(data_list)

def prepare_supervised_data(df):    # For SL agent
    df = df.dropna()
    X = df[config.FEATURES].values
    y = df["winner_A"].values
    return X, y

def save_data(df, filename):
    df.to_csv(os.path.join(config.data_path, filename), index=False)

def build_observation(row, p_hat=None, bankroll=None, only_features=False):     # For RL agent
    features = np.array([row[f] for f in config.FEATURES], dtype=np.float32)

    if only_features:
        return features

    edge = p_hat - (1.0 / row["odds_A"])

    obs = np.array([
        p_hat,
        row["odds_A"],
        edge,
        bankroll / 1000.0,
        row["lan"],
        row["bo_type"],
        row["ln_games_A"],
        row["ln_games_B"],
    ], dtype=np.float32)

    return obs
