from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated
import joblib
import pandas as pd

app=FastAPI()

pipeline = joblib.load("models/final_pipeline.pkl")

class Feature_Names(BaseModel):
    longitude:float|None = Field(ge=-125,le=-114)
    latitude:float|None = Field(ge=32,le=42)
    housing_median_age:float|None =Field(ge=0)
    total_rooms:float|None = Field(ge=0)
    total_bedrooms:float|None = Field(ge=0)
    population:float|None = Field(ge=0)
    households:float|None = Field(ge=0)
    median_income:float|None = Field(ge=0)
    ocean_proximity:str

@app.post("/predict")
def predict_amount(features: Feature_Names):

    input_data = pd.DataFrame([features.model_dump()])

    print(input_data)
    print(input_data.dtypes)

    prediction = pipeline.predict(input_data)

    print(prediction)

    return {
        "predicted_house_value": float(prediction[0])
    }