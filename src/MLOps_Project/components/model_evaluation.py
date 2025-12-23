import pandas as pd
import os
from src.MLOps_Project.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

class Model_Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        target = test_data[self.config.target_column]
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(columns=[self.config.target_column], axis=1)
        test_y = target

        mlflow.set_tracking_uri('https://dagshub.com/PAWANPAHUNE/MLOps_Project.mlflow')
        mlflow.set_experiment("MLOps_Project")

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="ELASTICNET_REG_MODEL"
        )
