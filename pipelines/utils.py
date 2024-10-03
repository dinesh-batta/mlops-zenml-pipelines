import logging

import pandas as pd
# from mlops_test.src.data_cleaning import DataCleaning,DataPreProcessStrategy
from src.data_cleaning import DataCleaning,DataPreProcessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv")
        df = df.sample(n=30)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["Default"], axis=1, inplace=True)
        print("test data: ",df.head().T)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
