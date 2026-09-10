import pandas as pd
import numpy as np
from preprocessing_model import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from xgboost import XGBRegressor
from  FE import FeatureEngineering
import joblib
from pathlib import Path


BASE_DIR=Path(__file__).resolve().parent.parent

DATA_PATH=BASE_DIR/'data'/'housing.csv'


data=pd.read_csv(DATA_PATH)

X,y=data.drop(columns=['median_house_value']) , data['median_house_value']


model=XGBRegressor(n_estimators= 500,
    learning_rate= 0.05,
    max_depth= 7,
    subsample=1,
    colsample_bytree=0.8
)

final_pipeline=make_pipeline(FeatureEngineering(),preprocessing,model)

final_pipeline.fit(X,y)

joblib.dump(final_pipeline,'models/final_pipeline.pkl')