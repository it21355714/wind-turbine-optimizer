import sys
import pandas as pd
from src.exception import CustomeException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomeException(e,sys)



class CustomData:
    def __init__(self, wind_speed, Rotor_speed, Nacelle_position, Blade_angle):
        self.wind_speed = wind_speed
        self.Rotor_speed = Rotor_speed
        self.Nacelle_position = Nacelle_position
        self.Blade_angle = Blade_angle
        
             


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Wind speed (m/s)": [self.wind_speed],
                "Rotor speed (RPM)": [self.Rotor_speed],
                "Nacelle position ": [self.Nacelle_position],
                "Blade angle": [self.Blade_angle]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomeException(e, sys)