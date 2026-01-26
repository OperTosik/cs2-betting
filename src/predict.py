import os
import numpy as np
import pandas as pd
import pickle
from agentSL import SupervisedAgent, SupervisedAgentCalibrated
from agentRL import BettingAgent
from utils import build_observation
from config import config

df = pd.read_csv(os.path.join(config.data_path, config.predict_path))

sl_agent = SupervisedAgent(os.path.join(config.models_path, config.catboost_model_name))
rl_agent = BettingAgent(os.path.join(config.models_path, config.pro_model_name))
slc_agent = SupervisedAgentCalibrated(os.path.join(config.models_path, "calibrated_" + config.catboost_model_name))

bankroll = 1000.0

for i, row in df.iterrows():
    p_hat = sl_agent.predict_single(
        build_observation(row, only_features=True)
    )    
    p_hat_cal = slc_agent.predict_single_calibrated(
        build_observation(row, only_features=True)
    )
    
    uncertainty = np.std([slc_agent.predict_single_calibrated(build_observation(row, only_features=True)) for _ in range(5)], axis=0)
    
    obs = build_observation(row, p_hat, p_hat_cal, uncertainty, bankroll)
    action = rl_agent.act(obs)

    if action > 0:
        map_match = row["map"]
        win_match = row["team_A"]
        print(f"Match {i}: P={p_hat:.3f}, Action={action}, Map={map_match}, Winner={win_match}")
    else:
        print(f"Match {i}: P={p_hat:.3f}, Action={action}")
