import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from agentSL import SupervisedAgent
from utils import build_observation


class CS2BettingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path,
        model_path,
        initial_bankroll: float = 1000.0
    ):
        super().__init__()
        self.df = pd.read_csv(data_path).dropna()
        self.sl_agent = SupervisedAgent(model_path)
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.idx = 0

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),         # size of obs
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
        p_hat = self.sl_agent.predict_single(
            build_observation(row, only_features=True)
        )
        return build_observation(row, p_hat, self.bankroll)

    def step(self, action):
        row = self.df.iloc[self.idx]
        odds = row["odds_A"]
        result = row["winner_A"]

        stake_frac = [0.0, 0.01, 0.02, 0.05][action]
        stake = stake_frac * self.bankroll

        reward = 0.0
        if stake > 0:
            if result == 1:
                reward = stake * (odds - 1)
            else:
                reward = -stake

        self.bankroll += reward
        self.idx += 1

        terminated = self.idx >= len(self.df) - 1 
        truncated = False
        done = terminated or truncated

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "bankroll": self.bankroll,
            "step": self.idx,
            # можно добавить odds, result и т.д. — полезно для логирования
        }

        return obs, reward, terminated, truncated, info