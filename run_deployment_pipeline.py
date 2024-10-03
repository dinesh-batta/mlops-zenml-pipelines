import click
from rich import print
from typing import cast
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# from mlops_test.pipelines.deployment_pipeline import continuous_deployment_pipeline ,inference_pipeline
from pipelines.deployment_pipeline import continuous_deployment_pipeline ,inference_pipeline



DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


# @click.command()
# @click.option(
#     "--config",
#     "-c",
#     type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
#     default=DEPLOY_AND_PREDICT,
#     help="Optionally you can choose to only run the deployment "
#     "pipeline to train and deploy a model (`deploy`), or to "
#     "only run a prediction against the deployed model "
#     "(`predict`). By default both will be run "
#     "(`deploy_and_predict`).",
# )
# @click.option(
#     "--min-accuracy",
#     default=0.92,
#     help="Minimum accuracy required to deploy the model",
# )

def run_deployment(config:str="deploy"):
    deploy = config== DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config== PREDICT or config == DEPLOY_AND_PREDICT
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    print("Deploy: ",deploy)
    print("deployer component: ",mlflow_model_deployer_component)
    if deploy:
        data_path = "/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv"
        continuous_deployment_pipeline(datapath= data_path,workers =1,timeout=60)
    if predict:
        inference_pipeline(pipeline_name = "continuous_deployment_pipeline",pipeline_step_name = "mlflow_model_deployer_step")


    print(
            "You can run:\n "
            f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
            "[/italic green]\n ...to inspect your experiment runs within the MLflow"
            " UI.\nYou can find your runs tracked within the "
            "`mlflow_example_pipeline` experiment. There you'll also be able to "
            "compare two or more runs.\n\n"
        )

    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model",
        )
    print("Existing services: ",existing_services)
    if existing_services:
            service = cast(MLFlowDeploymentService, existing_services[0])
            print("Service: ",service.is_stopped)
            if service.is_running:
                print(
                    f"The MLflow prediction server is running locally as a daemon "
                    f"process service and accepts inference requests at:\n"
                    f"    {service.prediction_url}\n"
                    f"To stop the service, run "
                    f"[italic green]`zenml model-deployer models delete "
                    f"{str(service.uuid)}`[/italic green]."
                )
            elif service.is_failed:
                print(
                    f"The MLflow prediction server is in a failed state:\n"
                    f" Last state: '{service.status.state.value}'\n"
                    f" Last error: '{service.status.last_error}'")
            elif service.is_stopped:
                print(
                    f"The MLflow prediction server is in a stopped state:\n"
                    f" Last state: '{service.status.state.value}'\n"
                    f" Last error: '{service.status.last_error}'\n"
                    f"service url {service.prediction_url}")
            else:
                print("Service status unknown !!")
    else:
            print(
                "No MLflow prediction server is currently running. The deployment "
                "pipeline must run first to train a model and deploy it. Execute "
                "the same command with the `--deploy` argument to deploy a model."
            )


if __name__ == "__main__":
    # deploy --> 1
    # predict --> 2
    flag = 1
    if flag == 1:
        run_deployment(config="deploy")
    if flag == 2:
        run_deployment(config="predict")
