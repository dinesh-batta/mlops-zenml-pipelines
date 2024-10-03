import logging
import pandas as pd
import numpy as np
from typing import Union
from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class defining strategy for handling data"""
    @abstractmethod
    def handle_data(self,df:pd.DataFrame)-> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """Data preprocessing step"""

    def handle_data(self,df:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:

            df = df.select_dtypes(include=[np.number])
            df.fillna({"Client_Income":df["Client_Income"].mean() ,"Loan_Annuity": df["Loan_Annuity"].mean(),"Age_Days":df["Age_Days"].mean()},inplace=True)
            df["total_family"] = df["Child_Count"]+ df["Client_Family_Members"]
            # df["Client_Income"].fillna(df["Client_Income"].mean(),inplace=True)
            # df["Loan_Annuity"].fillna(df["Loan_Annuity"].mean(),inplace=True)
            # df["Age_Days"].fillna(df["Age_Days"].mean(),inplace=True)
            cols_to_drop = []
            #df = df.drop(columns=cols_to_drop)
            df = df.drop(columns=['ID', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Score_Source_1', 'Child_Count',
                                  'Client_Family_Members','Population_Region_Relative','Mobile_Tag', 'Homephone_Tag',
                                  'Workphone_Working','Score_Source_3', 'Social_Circle_Default', 'Phone_Change',
                                    'Credit_Bureau'])
            df.fillna(df.mean(),inplace=True)
            print("Data preprocessed: ",df.info())
            return df
        except Exception as e:
            logging.error("Error Processing data:  {}".format(e))
            raise e

class DataDivideStragey(DataStrategy):
    """Data divide strategy"""
    def handle_data(self,df:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
            try:
                x = df.drop(columns="Default")
                y = df["Default"]
                x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=6)
                print("Data divided into test and train: ",x_train.shape)
                return x_train,x_test,y_train,y_test
            except Exception as e:
                logging.error("Error in dividing data into train and test {}".format(e))
                raise e

class DataCleaning():
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.df = data
        self.strategy = strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error("Error clenaing data {}".format(e))
            raise e

if __name__ == "__main__":
    data = pd.read_csv("/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv")
    data_cleaning = DataCleaning(data,DataPreProcessStrategy())
    cld_df = data_cleaning.handle_data()
    data_cleaning = DataCleaning(cld_df,DataDivideStragey())
    xtr,xte,ytr,yte = data_cleaning.handle_data()
