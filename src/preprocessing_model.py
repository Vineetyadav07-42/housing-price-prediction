import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector



numerical_pipeline =  make_pipeline(SimpleImputer(strategy='median')  ,  StandardScaler())

categorical_pipeline =  make_pipeline(SimpleImputer(strategy='most_frequent')  ,  OneHotEncoder())

preprocessing =  make_column_transformer((numerical_pipeline,make_column_selector(dtype_include=np.number)),
                                         (categorical_pipeline,make_column_selector(dtype_include=object)))


