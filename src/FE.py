from sklearn.base import BaseEstimator,TransformerMixin

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,data):
        data=data.copy()
        data['rooms_per_households']=data['total_rooms']/data['households']
        data['bedrooms_room_ratio']=data['total_bedrooms']/data['total_rooms']
        data['population_per_households']=data['population']/data['households']
        return data 