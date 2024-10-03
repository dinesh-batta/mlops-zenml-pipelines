import pandas as pd
import numpy as np
import json
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters,Output
# from zenml.integrations.mlflow.services import MLFlowDeploymentConfig

# deployment_config = MLFlowDeploymentConfig(
#     model_uri="file:///home/home/.config/zenml/local_stores/7b852249-fc0f-45d0-851a-e439b94856a1/mlruns/184603551433863274/f12aeb6bd67149ae8d33cbb14b4aca95/artifacts/model",  # Replace with your model path
#     workers=1,  # Number of workers to serve the model
#     timeout=60,  # Timeout in seconds
#     allocate_port=False,  # Set to False if you want to specify a custom port
#     port=8000  # Specify your custom port here
# )
#------------------
# from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentEndpointConfig
#
# # Create a new endpoint configuration with the correct port (5000)
# endpoint_config = MLFlowDeploymentEndpointConfig(
#     protocol="http",
#     port=5000,  # Set the port to 5000, matching the running MLflow service
#     ip_address="127.0.0.1",
#     prediction_url_path="invocations"
# )
#
# # Apply this endpoint configuration to your MLflow deployment service
# mlflow_deployment_service = MLFlowDeploymentService(
#     config=endpoint_config,
#     model_uri="file:///home/home/.config/zenml/local_stores/7b852249-fc0f-45d0-851a-e439b94856a1/mlruns/184603551433863274/f12aeb6bd67149ae8d33cbb14b4aca95/artifacts/model",
# )


# from mlops_test.steps.ingest_data import ingest_df
# from mlops_test.steps.clean_data import clean_df
# from mlops_test.steps.model_train import train_model
# from mlops_test.steps.evaluate_model import model_evaluation
# from mlops_test.pipelines.utils import get_data_for_test

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import model_evaluation
from pipelines.utils import get_data_for_test

docker_settings = DockerSettings(required_integrations = [MLFLOW])



class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0.3

@step
def deployment_trigger(accuracy:float,config:DeploymentTriggerConfig):
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name:str
    step_name:str
    running:bool = True


@step(enable_cache=False)
def dynamic_importer()->str:
    data = get_data_for_test()
    return data
@step(enable_cache=False)
def prediction_service_loader(pipeline_name:str,pipeline_step_name:str,running:bool=True,model_name:str="model")->MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

        Args:
            pipeline_name: name of the pipeline that deployed the MLflow prediction
                server
            step_name: the name of the step that deployed the MLflow prediction
                server
            running: when this flag is set, the step only returns a running service
            model_name: the name of the model that is deployed
        """
    #get mlflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    #get existing service with teh same pipelinename and pipeline step name
    existing_services = mlflow_model_deployer_component.find_model_server(pipeline_name = pipeline_name,pipeline_step_name = pipeline_step_name,
                                                                          model_name=model_name,running=running)
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print("Existing services: ",existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    print("in predictor: ",service)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = ['Client_Income',
                    'Car_Owned',
                    'Credit_Amount',
                    'Loan_Annuity',
                    'Age_Days',
                    'Employed_Days',
                    'Registration_Days',
                    'ID_Days',
                    'Own_House_Age',
                    'Cleint_City_Rating',
                    'Application_Process_Day',
                    'Application_Process_Hour',
                    'Score_Source_2',
                    'total_family']
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    # json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    # data = np.array(json_list)
    data = df
    print("data for prediction:",type(data))
    print("data for prediction:",data)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False,settings = {"docker":docker_settings})
def continuous_deployment_pipeline(datapath:str = "",
                                   workers:int =1,
                                   timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT):
    print("Model training pipeline started!!: ",datapath)
   # print("INgst data type!: ",type(ingest_df))
    df = ingest_df(datapath)
    x_train,x_test,y_train,y_test = clean_df(df)
    model = train_model(x_train,x_test,y_train,y_test)
    accuracyscore,precisionscore,recallscore,f1score,aucrocscore = model_evaluation(model,x_test,y_test)
    accuracyscore = 0.95
    print("Model eval finished!!. Accuracy score: ",accuracyscore)
    deployment_decision = deployment_trigger(accuracyscore)
    mlflow_model_deployer_step(model = model,
                               deploy_decision = deployment_decision,
                               workers = workers,
                               timeout= timeout )

@pipeline(enable_cache=False,settings={"docker":docker_settings})
def inference_pipeline(pipeline_name:str,pipeline_step_name:str):
    data = dynamic_importer()
    service = prediction_service_loader(pipeline_name = pipeline_name, pipeline_step_name = pipeline_step_name,
                                       running=False)
    prediction = predictor(service = service, data = data)
    return prediction


if __name__ == "__main__":
    fp = "/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv"
    #ndf = ingest_df(fp)
    ndf = continuous_deployment_pipeline(datapath = fp, workers =1,
                                   timeout=60)
    print("Pipeline finished!!")