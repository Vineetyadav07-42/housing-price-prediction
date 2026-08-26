import pandas as pd
import numpy as np
from src.preprocessing_model import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from xgboost import XGBRegressor
from  src.FE import FeatureEngineering
import joblib

data=pd.read_csv(r'C:\Users\Vineet\housing.csv')

X,y=data.drop(columns=['median_house_value']) , data['median_house_value']


model=XGBRegressor(n_estimators= 500,
    learning_rate= 0.05,
    max_depth= 7,
    subsample=1,
    colsample_bytree=0.8
)
final_pipeline=make_pipeline(FeatureEngineering(),preprocessing,model)
final_pipeline.fit(X,y)
joblib.dump(final_pipeline,'final_pipeline.pkl')