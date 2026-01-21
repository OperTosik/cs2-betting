import os
import pandas as pd
from catboost import CatBoostClassifier
from utils import prepare_supervised_data
from config import config


df = pd.read_csv(os.path.join(config.data_path, "commonData.csv"))
X, y = prepare_supervised_data(df)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=3000,
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=200
)

model.fit(
    X,
    y,
    # eval_set=(X_val, y_val),
    use_best_model=True
)

model.save_model("models/catboost_model.cbm")

print("✅ CatBoost model trained")
