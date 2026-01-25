import os
from stable_baselines3 import PPO
from env import CS2BettingEnv
from config import config

env = CS2BettingEnv(
    data_path=os.path.join(config.data_path, "commonData.csv"),
    model_path=os.path.join(config.models_path, config.catboost_model_name),
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
model.save(os.path.join(config.models_path, config.pro_model_name))

print("✅ PPO trained")
