from stable_baselines3 import PPO

class BettingAgent:
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)

    def act(self, obs):
        """
        Return action PPO
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)