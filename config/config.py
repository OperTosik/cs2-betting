data_path = "datasets"
predict_path = "predictDataReflected.csv"
models_path = "models"
result_path = "IEMCologneMajor2026.txt"

catboost_model_name = "catboost_model_v2_4.cbm"
pro_model_name = "pro_model_v2_4"

DATA = [
    "BlastBounty2026Season1Reflected.csv",
    "IEMKrakow2026Reflected.csv",
    "PGLClujNapoca2026Reflected.csv",
    "ESLProLeagueSeason23Reflected.csv",
    "BlastOpenRotterdam2026Reflected.csv",
    "PGLBucharest2026Reflected.csv",
    "IEMRio2026Reflected.csv",
    "PGLAstana2026Reflected.csv",
    "IEMAtlanta2026Reflected.csv",
    "CSAsiaChampionships2026Reflected.csv"
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
    "Train": 7,
    "Vertigo": 8
}