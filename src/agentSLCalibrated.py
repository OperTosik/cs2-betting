import os
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
# from agentSL import SupervisedAgent
from config import config
from utils import prepare_supervised_data

df = pd.read_csv(os.path.join(config.data_path, "commonData.csv"))
X, y = prepare_supervised_data(df)

sl_agent = CatBoostClassifier()
sl_agent.load_model(os.path.join(config.models_path, config.catboost_model_name))
# sl_agent.set_params(use_best_model=False)

calibrated_model = CalibratedClassifierCV(
    estimator=sl_agent,
    # method='isotonic',        # For a dataset with >2000-3000 samples
    method='sigmoid',           # For a dataset with <1000 samples
    cv=5
)

calibrated_model.fit(X, y)

with open(os.path.join(config.models_path, "calibrated_" + config.catboost_model_name), 'wb') as f:
    pickle.dump(calibrated_model, f)
    
print("✅ Calibrated CatBoost model saved")