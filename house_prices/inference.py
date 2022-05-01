import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import standardize_data, clean_data


MODEL_PATH = '../models/model.joblib'
FEATURES_CATEGORICAL = ['HouseStyle', 'BldgType']
FEATURES_CONTINUOUS = ['TotalBsmtSF', 'GrLivArea', 'GarageCars', 'GarageArea']
FEATURES = FEATURES_CATEGORICAL + FEATURES_CONTINUOUS


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    linear_model_loaded = joblib.load(MODEL_PATH)
    input_data = clean_data(input_data, FEATURES)
    # Identifying feature variables
    X = input_data.copy()[FEATURES]
    # Preprocessing data
    X = standardize_data(FEATURES_CATEGORICAL, FEATURES_CONTINUOUS, X)
    y_pred = linear_model_loaded.predict(X)
    return y_pred
