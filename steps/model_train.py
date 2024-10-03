import pandas as pd
import logging
import mlflow
from zenml import step
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig
#from mlops_test.steps.config import ModelNameConfig
# from mlops_test.src.model_dev import LogisticRegressionModel,LinearSVMModel
from src.model_dev import LogisticRegressionModel,LinearSVMModel

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

print("experiment_tracker: ",Client().active_stack.experiment_tracker.name)

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train:pd.DataFrame,x_test:pd.DataFrame,
                y_train:pd.Series,y_test:pd.Series,
                config:ModelNameConfig)->ClassifierMixin: #x_test:pd.DataFrame,y_test:pd.DataFrame,
    try:
        model = None
        print("IN train model step")
        if config.model_name == "LogisticRegression":

            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
            trained_model = model.train_model(x_train,x_test,y_train,y_test)
            logging.info("Model build successful: {}".format(config.model_name))
            return trained_model
        elif config.model_name == "LinearSVM":
            mlflow.sklearn.autolog()
            model = LinearSVMModel()
            trained_model = model.train_model(x_train,x_test,y_train,y_test)
            logging.info("Model build successful: {}".format(config.model_name))
            return trained_model
        else:
            raise ValueError("Model {} not supported!!".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model!! {}".format(e))
        raise e
