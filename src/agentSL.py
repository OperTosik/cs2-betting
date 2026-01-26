import pickle
import numpy as np
from catboost import CatBoostClassifier
# from sklearn.model_selection import KFold

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
    
class SupervisedAgentCalibrated:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        # print(f"Model classes: {self.model.classes_}")
        # print(f"Model classes type: {type(self.model.classes_)}")
        
    def load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    
    def predict_single_calibrated(self, x_row):
        return self.model.predict_proba(x_row.reshape(1, -1))[0][1]
    
        