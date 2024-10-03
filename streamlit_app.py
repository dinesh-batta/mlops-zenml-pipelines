import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# from mlops_test.run_deployment_pipeline import run_deployment
# from mlops_test.pipelines.deployment_pipeline import prediction_service_loader
from pipelines.deployment_pipeline import prediction_service_loader
print("loaded predictoin service loader")
from run_deployment_pipeline import run_deployment


def main():
    st.title("End to End Loan default Pipeline with ZenML")

    high_level_image = Image.open("/home/home/PycharmProjects/pocs/mlops_test/_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("/home/home/PycharmProjects/pocs/mlops_test/_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )

    #payment_installments = st.sidebar.slider("Payment Installments")
    #car_owned = st.number_input("Car_Owned")
    client_income = st.number_input("Client_Income")
    car_owned = st.sidebar.slider("Car_Owned")
    loan_amount = st.number_input("Credit_Amount")
    emi = st.number_input("Loan_Annuity")
    age_days = st.number_input("Age_Days")
    employed_days = st.number_input("Employed_Days")
    id_days = st.number_input("ID_Days")
    own_house_age = st.number_input("Own_House_Age")
    client_city_rating = st.number_input("Cleint_City_Rating")
    process_date = st.number_input("Application_Process_Day")
    process_hour = st.number_input("Application_Process_Hour")
    score2 = st.number_input("Score_Source_2")
    total_family = st.number_input("total_family")

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_deployment()

        df = pd.DataFrame(
            {    "Client_Income":[client_income],
                "Car_Owned":[car_owned],
                "Credit_Amount":[loan_amount],
                "Loan_Annuity":[emi],
                "Age_Days":[age_days],
                "Employed_Days":[employed_days],
                "ID_Days":[id_days],
                "Own_House_Age":[own_house_age],
                "Cleint_City_Rating":[client_city_rating],
                "Application_Process_Day":[process_date],
                "Application_Process_Hour":[process_hour],
                "Score_Source_2":[score2],
                "total_family":[total_family]}
            # {
            #     "payment_sequential": [payment_sequential],
            #     "payment_installments": [payment_installments],
            #     "payment_value": [payment_value],
            #     "price": [price],
            #     "freight_value": [freight_value],
            #     "product_name_lenght": [product_name_length],
            #     "product_description_lenght": [product_description_length],
            #     "product_photos_qty": [product_photos_qty],
            #     "product_weight_g": [product_weight_g],
            #     "product_length_cm": [product_length_cm],
            #     "product_height_cm": [product_height_cm],
            #     "product_width_cm": [product_width_cm],
            # }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                pred
            )
        )
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("/home/home/PycharmProjects/pocs/mlops_test/_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()