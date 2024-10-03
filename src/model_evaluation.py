import logging
import numpy as np
from abc import ABC,abstractmethod
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,f1_score

class Evaluation(ABC):
    """Abstract class for evaluation"""
    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):

        pass



class accuracy(Evaluation):
    # def __int__(self,y_true,y_predict):
    #     self.y_true = y_true
    #     self.y_predict = y_predict

    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):
        try:
            logging.info("Calculating accuracy score!!")
            score = accuracy_score(y_true,y_predict)
            print("Accuracy score: ",score)
            logging.info("Accuracy score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error calculating accuracy score : {}".format(e))
            raise e

class precision(Evaluation):
    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):
        try:
            logging.info("Calculating precision score!!")
            score = precision_score(y_true,y_predict)
            logging.info("Precision score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error calculating precision score: {}".format(e))
            raise e

class recall(Evaluation):
    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):
        try:
            logging.info("Calculating recall score!!")
            score = recall_score(y_true,y_predict)
            logging.info("Recall score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error calculating recall score: {}".format(e))
            raise e

class auc_roc(Evaluation):
    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):
        try:
            logging.info("Calculating AUC-ROC score!!")
            score = roc_auc_score(y_true,y_predict)
            logging.info("AUC-ROC score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error calculating AUC-ROC score: {}".format(e))
            raise e
class f1(Evaluation):
    def calcualte_score(self,y_true:np.ndarray,y_predict:np.ndarray):
        try:
            logging.info("Calculating f1 score!!")
            score = f1_score(y_true,y_predict)
            logging.info("f1 score score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error calculating f1 score: {}".format(e))
            raise e