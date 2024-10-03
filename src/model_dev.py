import logging
import pandas as pd

from abc import ABC,abstractmethod
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class Model(ABC):

    @abstractmethod
    def train_model(self,x_train,x_test,y_train,y_test):
        pass
    @abstractmethod
    def optimize(self,x_train,x_test,y_train,y_test):
        pass

class LogisticRegressionModel(Model):
    # def __init__(self,xtrain,ytrain):
    #     self.x_train = xtrain
    #     self.y_train = ytrain

    def train_model(self,x_train,x_test,y_train,y_test,**kwargs) :
        try:
            model = LogisticRegression(**kwargs)
            model.fit(x_train,y_train)
            logging.info("Model training complete!!")
            return model
        except Exception as e:
            logging.error("Error training Logistic regression: {}".format(e))


    def optimize(self, x_train,x_test,y_train,y_test,**kwargs):
        try:
            # Define the hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],  # Regularization strength
                'penalty': ['l1', 'l2'],  # Regularization type (L1 or L2)
                'solver': ['liblinear']  # Solver that supports L1 regularization
            }
            model = LogisticRegression(**kwargs)
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,
                                       scoring='accuracy')
            # Fit the model to find the best parameters
            grid_search.fit(x_train, y_train)

            # Print the best parameters and the best score
            print("Best Parameters:", grid_search.best_params_)
            print("Best Accuracy:", grid_search.best_score_)

            # Predict on the test set using the best model
            y_pred = grid_search.best_estimator_.predict(x_test)
            # Evaluate the model
            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            return grid_search.best_params_
        except  Exception as e:
            logging.error("Error in parameter turing: {}".format(e))
            raise e

class LinearSVMModel(Model):
    # def __int__(self,xtrain,ytrain):
    #     self.x_train = xtrain
    #     self.y_train = ytrain

    def train_model(self,x_train,x_test,y_train,y_test,**kwargs) :
        try:
            model = LinearSVC(**kwargs)
            model.fit(x_train,y_train)
            logging.info("Model training complete!!")
            return model
        except Exception as e:
            logging.error("Error training SVM moel: {}".format(e))
            raise e

    def optimize(self, x_train,x_test,y_train,y_test,**kwargs):
        try:
            # Define the hyperparameter grid
            param_grid = {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
                        'loss': ['hinge', 'squared_hinge'],    # Type of loss function
                    }
            model = LinearSVC(**kwargs)
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2,
                                       scoring='accuracy')
            # Fit the model to find the best parameters
            grid_search.fit(x_train, y_train)

            # Print the best parameters and the best score
            print("Best Parameters:", grid_search.best_params_)
            print("Best Accuracy:", grid_search.best_score_)

            # Predict on the test set using the best model
            y_pred = grid_search.best_estimator_.predict(x_test)
            # Evaluate the model
            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            return grid_search.best_params_
        except  Exception as e:
            logging.error("Error in parameter turing: {}".format(e))
            raise e



class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train,x_test,y_train,y_test,param_grid):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.param_grid = param_grid

    def optimize(self, n_trials=100):
        # # Define the parameter grid to search
        # param_grid = {
        #     'n_estimators': [50, 100, 200],  # Number of trees in the forest
        #     'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
        #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        #     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
        # }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

        # Fit the model to find the best parameters
        grid_search.fit(self.x_train, self.y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Accuracy:", grid_search.best_score_)

        # Predict on the test set using the best model
        y_pred = grid_search.best_estimator_.predict(self.x_test)
        # Evaluate the model
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        return grid_search.best_params_



