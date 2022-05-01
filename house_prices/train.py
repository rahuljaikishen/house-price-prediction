import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from house_prices.preprocess import standardize_data, fit_encoder
from house_prices.preprocess import fit_scaler, clean_data


MODEL_PATH = '../models/model.joblib'
FEATURES_CATEGORICAL = ['HouseStyle', 'BldgType']
FEATURES_CONTINUOUS = ['TotalBsmtSF', 'GrLivArea', 'GarageCars', 'GarageArea']
FEATURES = FEATURES_CATEGORICAL + FEATURES_CONTINUOUS
TARGET = ['SalePrice']
FEATURES_AND_TARGET = FEATURES + TARGET


def build_model(data: pd.DataFrame) -> dict[str, str]:
    # Returns a dictionary with the model performances
    data.set_index('Id')
    data = clean_data(data, FEATURES_AND_TARGET)
    # Identifying feature and target variables
    X = data[FEATURES]
    y = data[TARGET]
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)
    # Creating and training the model
    train_model(X_train, y_train)
    # Evaluating model performance
    score = evaluate_model(X_test, y_test)
    return {'rsme': score}


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    # Fitting data
    fit_encoder(X_train[FEATURES_CATEGORICAL])
    fit_scaler(X_train[FEATURES_CONTINUOUS])
    # Transforming train data
    X_train = standardize_data(FEATURES_CATEGORICAL, FEATURES_CONTINUOUS,
                               X_train)
    # Fitting and training a linear regression model
    reg_multiple = LinearRegression()
    reg_multiple.fit(X_train, y_train)
    joblib.dump(reg_multiple, MODEL_PATH)


def evaluate_model(X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    # Transforming test data
    X_test = standardize_data(FEATURES_CATEGORICAL, FEATURES_CONTINUOUS,
                              X_test)
    # Calculating model performance
    reg_multiple = joblib.load(MODEL_PATH)
    y_pred = reg_multiple.predict(X_test)
    score = compute_rmsle(y_test, y_pred)
    return score


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
