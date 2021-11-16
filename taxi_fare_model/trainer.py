from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from taxi_fare_model.encoders import DistanceTransformer, TimeFeaturesEncoder
import numpy as np
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
from taxi_fare_model.data import get_data, clean_data


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[JP] [Tokyo] [jkclaar] TaxiFareModel + 1.0"
        self.mlflow_uri = "https://mlflow.lewagon.co/"


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        self.mlflow_log_param('model', 'linear')
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        ret = np.sqrt(((y_pred - y_test)**2).mean())
        self.mlflow_log_metric('rmse', ret)
        return ret


    @memoized_property
    def mlflow_client(self):

        mlflow.set_tracking_uri(self.mlflow_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":
    data = get_data()
    data = clean_data(data)
    X = data.drop(columns=['fare_amount'])
    y = data.fare_amount
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.evaluate(X_test, y_test)
