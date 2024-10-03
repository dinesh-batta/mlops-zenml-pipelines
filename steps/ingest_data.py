import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self,datapath:str):
        """Initialize the data ingestion class."""
        self.data_path = datapath

    def get_data(self) -> pd.DataFrame:
        # df = pd.read_csv("/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv")
        logging.info(f"Ingesting data from {self.data_path}")
        df = pd.read_csv(self.data_path,low_memory=False)
        print("Data read!!",df.head().T)
        return df

@step
def ingest_df(datapath:str) -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        print("in ingtest data-->")
        ingest_data = IngestData(datapath)

        df = ingest_data.get_data()
        print("in ingtest data--> data read completed!!")
        return df
    except Exception as e:
        print("Error while loading data!!")
        logging.error(f"Error while reading data {e}")
        raise e