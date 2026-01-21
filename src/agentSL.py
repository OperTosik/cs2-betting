import numpy as np
from catboost import CatBoostClassifier

class SupervisedAgent:
    def __init__(self, model_path: str):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def predict_proba(self, X):
        """
        Return probably of win team_A
        """
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, x_row):
        """
        For one string of data
        """
        return self.predict_proba(x_row.reshape(1, -1))[0]