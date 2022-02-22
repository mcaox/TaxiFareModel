# imports
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

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceToCenterTransformer
from TaxiFareModel.utils import compute_rmse

MLFLOW_URI = "https://mlflow.lewagon.co/"


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
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
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

    def save_model(self):
        """ Save the trained model into a model.joblib file """

        joblib.dump(self.pipeline, 'model.joblib')

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


TARGET = "fare_amount"
def main(src = None,uri=MLFLOW_URI):
    if src is None or not Path(src).exists():
        df = get_data()
    else:
        df = get_data(src=src)
    df = clean_data(df)

    y=df[TARGET]
    X=df.drop(columns=TARGET)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    scores = []
    for estimator,params in [
        (XGBRFRegressor(),{}),
        ]:
        print(estimator.__class__.__name__,params)
        trainer = Trainer(X_train,y_train,estimator,params,experiment_name="[FR] [Lyon] [mcaox] TaxiFareModel with dist_to_center")
        trainer.mlflow_uri = uri
        trainer.run()
        trainer.save_model()
        cv = cross_validate(trainer.pipeline,X_train,y_train,cv=5,scoring=make_scorer(compute_rmse, greater_is_better=False))
        score = cv.get("test_score").mean()
        scores.append(score)
        trainer.mlflow_log_metric("rmse",score)
        trainer.mlflow_log_param("model",trainer.pipeline[-1].__class__.__name__)
        for k,v in params.items():
            trainer.mlflow_log_param(k,v)
    return trainer,min(scores)

if __name__ == "__main__":
    src = Path(__file__).parents[1] / "raw_data" / "train.csv"
    trainer,score = main(src=src,uri="")
    print(score)