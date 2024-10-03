import logging
import mlflow
import pandas as pd
from zenml import step
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from typing import Tuple

# from mlops_test.src.model_evaluation import accuracy,precision,recall,f1,auc_roc
from src.model_evaluation import accuracy,precision,recall,f1,auc_roc


from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker = experiment_tracker.name)
def model_evaluation(model:ClassifierMixin,x_test:pd.DataFrame,y_test:pd.Series)-> \
        Tuple[Annotated[float,"accuracyscore"],
                Annotated[float,"precisionscore"],Annotated[float,"recallscore"],
                Annotated[float,"f1score"],Annotated[float,"aucrocscore"]] :
    try:
        y_pred = model.predict(x_test)
        accuracyscore = accuracy().calcualte_score(y_test,y_pred)
        mlflow.log_metric("accuracy_score",accuracyscore)
        precisionscore = precision().calcualte_score(y_test,y_pred)
        mlflow.log_metric("precision_score",precisionscore)
        recallscore = recall().calcualte_score(y_test,y_pred)
        mlflow.log_metric("recall_score",recallscore)
        f1score = f1().calcualte_score(y_test,y_pred)
        mlflow.log_metric("f1_score",f1score)
        aucrocscore = auc_roc().calcualte_score(y_test,y_pred)
        mlflow.log_metric("auc_roc_score",aucrocscore)
        print("accuracy score: ",accuracyscore)
        return accuracyscore,precisionscore,recallscore,f1score,aucrocscore
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e