# imports
import sys
from argparse import ArgumentParser
from pathlib import Path

import joblib
import mlflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRFRegressor


from google.cloud import storage

from TaxiFareModel.config import BUCKET_NAME, STORAGE_LOCATION, MLFLOW_URI, BUCKET_TRAIN_DATA_PATH
from TaxiFareModel.data import get_data, clean_data, get_X_y
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceToCenterTransformer, \
    OptimizingTransformer
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y,estimator=Lasso(),params={},experiment_name = "[FR] [Lyon] [mcaox] TaxiFareModel"):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.estimator = estimator
        self.params = params
        self.X = X
        self.y = y
        self._mlflow_uri = None
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        time_pipe = Pipeline(
            [
                ("timetrans",TimeFeaturesEncoder('pickup_datetime')),
                ('ohe', OneHotEncoder(handle_unknown='ignore',sparse=False))
            ]
        )
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        dist_pipe_to_center = Pipeline(
            [
                ("dist_trans",DistanceToCenterTransformer()),
                ('stdscaler', StandardScaler())
            ]
        )

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('distance_to_center', dist_pipe_to_center, ["pickup_latitude", "pickup_longitude"]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")


        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('optmiser',OptimizingTransformer()),
            ('model', self.estimator)
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.set_params(**self.params)
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def save_model(self,name='model.joblib'):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, name)
        return name

    @property
    def mlflow_uri(self):
        return self._mlflow_uri

    @mlflow_uri.setter
    def mlflow_uri(self,uri):
        self._mlflow_uri = uri

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def upload_model_to_gcp(self,name='model.joblib'):
        print("salut")
        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION + name)

        blob.upload_from_filename(name)


TARGET = "fare_amount"
def train_locally(src = None, uri=None):
    if src is None or not Path(src).exists():
        df = get_data()
    else:
        df = get_data(src=src)
    if uri is None:
        uri = MLFLOW_URI
    df = clean_data(df)
    X,y = get_X_y(df,TARGET,[TARGET])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    scores = []
    for estimator,params in [
        (XGBRFRegressor(),{}),
        ]:
        print(estimator.__class__.__name__,params)
        trainer = Trainer(X_train,y_train,estimator,params,experiment_name="[FR] [Lyon] [mcaox] TaxiFareModel with dist_to_center")
        trainer.mlflow_uri = uri
        trainer.run()
        trainer.save_model(f'model{estimator.__class__.__name__}.joblib')
        cv = cross_validate(trainer.pipeline,X_train,y_train,cv=5,scoring=make_scorer(compute_rmse, greater_is_better=False))
        score = cv.get("test_score").mean()
        scores.append(score)
        trainer.mlflow_log_metric("rmse",score)
        trainer.mlflow_log_param("model",trainer.pipeline[-1].__class__.__name__)
        for k,v in params.items():
            trainer.mlflow_log_param(k,v)
    return trainer,min(scores)


def train_on_gcloud(uri=None):
    if uri is None:
        uri = MLFLOW_URI
    src = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = get_data(src=src)
    df = clean_data(df)
    X,y = get_X_y(df,TARGET,[TARGET])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    estimator = XGBRFRegressor()
    params = {}
    trainer = Trainer(X_train,y_train,estimator,params,experiment_name="[FR] [Lyon] [mcaox] TaxiFareModel with dist_to_center gcloud")
    trainer.mlflow_uri = uri
    trainer.run()
    score = trainer.evaluate(X_test,y_test)
    trainer.mlflow_log_metric("rmse",score)
    trainer.mlflow_log_param("model",trainer.pipeline[-1].__class__.__name__)
    for k,v in params.items():
        trainer.mlflow_log_param(k,v)
    name = trainer.save_model(f'model{estimator.__class__.__name__}.joblib')
    trainer.upload_model_to_gcp(name)
    return trainer,score

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] =="gcloud":
        trainer,score = train_on_gcloud()
    else:
        src = Path(__file__).parents[1] / "raw_data" / "train.csv"
        trainer,score = train_locally(src=src, uri="")
    print(score)
