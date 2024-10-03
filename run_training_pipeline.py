from mlops_test.pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    #mlflow ui --backend-store-uri "file:/home/home/.config/zenml/local_stores/7b852249-fc0f-45d0-851a-e439b94856a1/mlruns"
    data_path = "/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv"
    train_pipeline(datapath = data_path)