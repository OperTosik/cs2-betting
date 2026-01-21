import os
from stable_baselines3 import PPO
from env import CS2BettingEnv
from config import config

env = CS2BettingEnv(
    data_path=os.path.join(config.data_path, "commonData.csv"),
    model_path=os.path.join(config.models_path, "catboost_model.cbm"),
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=256,
    verbose=1
)

model.learn(total_timesteps=300_000)
model.save("models/pro_model")

print("✅ PPO trained")
