import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


ENCODER_PATH = '../models/encoder.joblib'
SCALER_PATH = '../models/scaler.joblib'


def clean_data(data: pd.DataFrame, col_names: list) -> pd.DataFrame:
    # function to clean missing data
    data.dropna(subset=col_names, how='any', inplace=True)
    return data


def fit_encoder(categorical_data: pd.DataFrame) -> None:
    # fit encoder and create copy in the models folder
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(categorical_data)
    joblib.dump(encoder, ENCODER_PATH)


def fit_scaler(continuous_data: pd.DataFrame) -> None:
    # fit encoder and create copy in the models folder
    scaler = StandardScaler()
    scaler.fit(continuous_data)
    joblib.dump(scaler, SCALER_PATH)


def standardize_data(categorical_features: list, continuous_features: list,
                     data: pd.DataFrame) -> pd.DataFrame:
    data = transform_column(continuous_features, data)
    data = transform_categories(categorical_features, data)
    return data


def transform_categories(col_names: list, data: pd.DataFrame) -> pd.DataFrame:
    # Encodes categorical features of the dataset using OneHotEncoder
    encoder = joblib.load(ENCODER_PATH)
    categorical_encoded = encoder.transform(data[col_names])
    feature_names = encoder.get_feature_names_out(input_features=col_names)
    # Creating dataframe with encoded columns created
    categorical_encoded = pd.DataFrame(categorical_encoded,
                                       columns=feature_names, index=data.index)
    # Joining additional columns back into original dataset
    encoded_data = data.join(categorical_encoded)
    # Dropping categorically encoded data
    encoded_data.drop(columns=col_names, axis=1, inplace=True)
    return encoded_data


def transform_column(col_names: list, data: pd.DataFrame) -> pd.DataFrame:
    # Scales continuos features of a dataset using StandardScaler
    scaler = joblib.load(SCALER_PATH)
    standarized_data = scaler.transform(data[col_names])
    data.loc[:, col_names] = standarized_data
    return data
