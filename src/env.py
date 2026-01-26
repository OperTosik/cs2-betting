import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from agentSL import SupervisedAgent, SupervisedAgentCalibrated
from utils import build_observation


class CS2BettingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path,
        sl_model_path,
        slc_model_path,
        initial_bankroll: float = 1000.0
    ):
        super().__init__()
        self.df = pd.read_csv(data_path).dropna()
        self.sl_agent = SupervisedAgent(sl_model_path)
        self.slc_agent = SupervisedAgentCalibrated(slc_model_path)
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.idx = 0

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),         # size of obs
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.bankroll = self.initial_bankroll
        self.idx = 0

        obs = self._get_obs()
        info = {"bankroll": self.bankroll}

        return obs, info

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        
        # SL features
        sl_features = build_observation(row, only_features=True)

        # SL predictions
        p_hat = self.sl_agent.predict_single(sl_features)
        p_hat_cal = self.slc_agent.predict_single_calibrated(sl_features)
        probas = np.array([self.slc_agent.predict_proba(sl_features.reshape(1, -1)) for _ in range(5)])
        
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = probas.flatten()
        
        uncertainty = np.std(probas)

        obs = build_observation(
            row=row,
            p_hat=p_hat,
            p_hat_cal=p_hat_cal,
            uncertainty=uncertainty,
            bankroll=self.bankroll
        )

        return obs

    def step(self, action):
        row = self.df.iloc[self.idx]
        odds = row["odds_A"]
        result = row["winner_A"]

        stake_frac = [0.0, 0.01, 0.02, 0.05][action]
        stake = stake_frac * self.bankroll

        profit = 0.0
        if stake > 0:
            if result == 1:
                profit = stake * (odds - 1)
            else:
                profit = -stake

        # reward shaping
        reward = np.log((self.bankroll + profit) / self.bankroll) if stake > 0 else 0.0

        self.bankroll += profit
        self.idx += 1

        terminated = self.idx >= len(self.df) - 1
        truncated = False

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "bankroll": self.bankroll,
            "step": self.idx
        }

        return obs, reward, terminated, truncated, info
