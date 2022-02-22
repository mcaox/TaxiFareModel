# imports
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")


        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', Lasso())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

TARGET = "fare_amount"
def main(src = None):
    if src is None or not Path(src).exists():
        df = get_data()
    else:
        df = get_data(src=src)
    df = clean_data(df)
    y=df[TARGET]
    X=df.drop(columns=TARGET)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    trainer = Trainer(X_train,y_train)
    trainer.run()
    score = trainer.evaluate(X_test,y_test)
    return trainer.pipeline,score

if __name__ == "__main__":
    src = Path(__file__).parents[1] / "raw_data" / "train.csv"
    pipe,score = main(src=src)
    print(score)
