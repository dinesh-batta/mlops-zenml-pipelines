import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

# from mlops_test.src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStragey
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStragey
@step
def clean_df(df:pd.DataFrame)-> Tuple[  Annotated[pd.DataFrame,"x_train"],
                                        Annotated[pd.DataFrame,"x_test"],
                                        Annotated[pd.Series,"y_train"],
                                        Annotated[pd.Series,"y_test"] ]:
    """ Cleans the data and divides it into test and train"""
    try:
        process_strategy = DataPreProcessStrategy()
        data_clean = DataCleaning(df,process_strategy)
        cld_df = data_clean.handle_data()
        print("Data cols: ",cld_df.columns)
        divide_strategy = DataDivideStragey()
        data_clean = DataCleaning(cld_df,divide_strategy)
        x_train,x_test,y_train,y_test = data_clean.handle_data()
        logging.info("Data cleaning completed!!")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error("Error cleaning data: {} ".format(e))
        raise e