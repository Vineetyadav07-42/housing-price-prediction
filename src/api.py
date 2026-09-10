from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
import pandas as pd
from pathlib import Path


app=FastAPI()

BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR/'models'/'final_pipeline.pkl'


pipeline = joblib.load(MODEL_DIR)


class FeatureNames(BaseModel):
    longitude:float = Field(ge=-125,le=-114)
    latitude:float = Field(ge=32,le=42)
    housing_median_age:float = Field(ge=0)
    total_rooms:float = Field(ge=0)
    total_bedrooms:float = Field(ge=0)
    population:float = Field(ge=0)
    households:float = Field(ge=0)
    median_income:float = Field(ge=0)
    ocean_proximity:str 


@app.get('/')
def return_item():
    return {'message':'California House Price Prediction API is running'}


@app.post("/predict")
def predict_amount(features: FeatureNames):

    input_data = pd.DataFrame([features.model_dump()])

    prediction = pipeline.predict(input_data)

    print(prediction)

    return {
        "predicted_house_value": float(prediction[0])
    }