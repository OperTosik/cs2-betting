data_path = "datasets"
predict_path = "predictData.csv"
models_path = "models"

catboost_model_name = "catboost_model_v1_0.cbm"
pro_model_name = "pro_model_v1_0"

DATA = [
    "BlastBounty2026Season1Reflected.csv",
    "IEMKrakow2026.csv"
]

FEATURES = [
    "odds_A",
    "odds_B",
    "hltv_points_diff",
    "valve_points_diff",
    "form_A",
    "form_B",
    "map_wr_A",
    "map_wr_B",
    "h2h_A",
    "avg_rating_3_0_diff",
    "avg_rating_diff",
    "coach_wr_A",
    "coach_wr_B",
    "pick_a_winner_A",
    "roster_changed_last_30d_A",
    "roster_changed_last_30d_B"
]

MAPS = {
    "Dust2": 0,
    "Mirage": 1,
    "Inferno": 2,
    "Nuke": 3,
    "Overpass": 4,
    "Ancient": 5,
    "Anubis": 6,
    "Train": 7
}