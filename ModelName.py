from enum import Enum


class ModelName(str, Enum):
    default = ""
    Catboost = "cat"
    LightGBM = "lgb"
    XGBoost = "xgb"
    GBoost = "gboost"
    Linear_SVC = "LinearSVC"
    SVC = "SVC"
    KNN = "KNN"
    Logistic_regression = "LR"
    Random_forest = "RF"
    Decision_tree = "DT"