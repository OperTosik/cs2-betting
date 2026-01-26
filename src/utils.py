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

def build_observation(     # For RL agent
    row, 
    p_hat=None, 
    p_hat_cal=None, 
    uncertainty=None,
    bankroll=None, 
    only_features=False, 
):
    features = np.array(
        [row[f] for f in config.FEATURES],
        dtype=np.float32
    )

    if only_features:
        return features
    
    # Safety checks
    assert p_hat is not None
    assert bankroll is not None
    assert p_hat_cal is not None
    assert uncertainty is not None

    edge = p_hat_cal - (1.0 / row["odds_A"])
    
    obs = np.array([        # If you change features, you have to change 32 line in env.py
        p_hat,
        p_hat_cal,
        uncertainty,
        row["odds_A"],
        edge,
        bankroll / 1000.0,
        config.MAPS[row["map"]],
        row["lan"],
        row["bo_type"],
        row["ln_games_A"],
        row["ln_games_B"],
    ], dtype=np.float32)

    return obs

