from zenml import pipeline
from mlops_test.steps.ingest_data import ingest_df
from mlops_test.steps.clean_data import clean_df
from mlops_test.steps.model_train import train_model
from mlops_test.steps.evaluate_model import model_evaluation


@pipeline(enable_cache=False)
def train_pipeline(datapath:str):
    print("ingest df type: ",type(ingest_df))
    df = ingest_df(datapath)
    x_train,x_test,y_train,y_test = clean_df(df)
    #print("Input columns: ",x_train.shape)
    model = train_model(x_train,x_test,y_train,y_test) #,x_test,y_test
    accuracyscore,precisionscore,recallscore,f1score,aucrocscore = model_evaluation(model,x_test,y_test)
    print("scores: aucroc",str(aucrocscore)," f1 score: ",str(f1score))


if __name__ == "__main__":
    fp = "/home/home/PycharmProjects/pocs/mlops_test/data/Dataset_small_final.csv"
    ndf = train_pipeline(datapath = fp)
    # print("data read: ",ndf.shape)