from zenml.steps import  BaseParameters

class ModelNameConfig(BaseParameters):
    model_name:str = "LogisticRegression"
    finetune : bool  = False

    class Config:
        protected_namespaces = ()