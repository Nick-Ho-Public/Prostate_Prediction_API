import joblib
import os
import numpy as np
from ModelName import ModelName


def prostate_bin(prostate, modelName):
    if not modelName:
        modelName = ModelName.Catboost
    model = joblib.load(os.sep.join(["models", "bin", modelName.value + ".joblib"]))
    prostate = np.array(list(prostate.dict().values())).reshape(1, -1)
    pred = model.predict(prostate)
    proba = model.predict_proba(prostate)[0][1]
    return {"Prediction": bool(pred), "Probability": float(proba)}


def prostate_sign(prostate, modelName):
    if not modelName:
        modelName = ModelName.Catboost
    model = joblib.load(os.sep.join(["models", "sign", modelName.value + ".joblib"]))
    prostate = np.array(list(prostate.dict().values())).reshape(1, -1)
    pred = model.predict(prostate)
    proba = model.predict_proba(prostate)[0][1]
    return {"Prediction": bool(pred), "Probability": float(proba)}