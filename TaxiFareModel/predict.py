import os
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from google.cloud import storage

from sklearn.metrics import mean_absolute_error, mean_squared_error

from TaxiFareModel.config import BUCKET_NAME, STORAGE_LOCATION

PATH_TO_LOCAL_MODEL = 'model.joblib'

def get_test_data(path):
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    print(path)
    df = pd.read_csv(path)
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def download_model(model_name="model.joblib", bucket=BUCKET_NAME):
    client = storage.Client().bucket(bucket)
    blob = client.blob(f'{STORAGE_LOCATION}{model_name}')
    path_d = Path("tmp")
    path_d.mkdir(exist_ok=True)
    path = path_d / model_name
    blob.download_to_filename(path)
    return path


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(path_test_data="raw_data",
                            path_trained_model="model.joblib",
                            kaggle_upload=False):
    df_test = get_test_data(path=path_test_data)
    pipeline = get_model(path_trained_model)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = Path(path_test_data.parent) / f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name.stem
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':
    path_test_data = Path("raw_data") / "test.csv"
    path_model = download_model(model_name="modelXGBRFRegressor.joblib",
                          bucket=BUCKET_NAME)
    generate_submission_csv(path_test_data=path_test_data,
                            path_trained_model=path_model,
                            kaggle_upload=False)
    path_model.unlink(missing_ok=True)
