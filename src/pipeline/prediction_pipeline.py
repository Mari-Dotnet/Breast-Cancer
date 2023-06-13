import pandas as pd
import os,sys
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class predic_pipeline:
    def __init__(self) -> None:
        pass

    def Predictvalues(self,features):
        try:
            logging.info("Prediction started")
            pre_processor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model_processor.pkl')

            logging.info('preprocessor path loaded')
            print("below processor object load")
            pre_processor=load_object(pre_processor_path)
            model=load_object(model_path)
            print("After processor object load")
            logging.info(f"features values{features}")
            data_sclaed=pre_processor.transform(features)   
            print('features',data_sclaed)
            print("model",model)
            predict=model.predict(data_sclaed)
            return predict
        except Exception as ex:
            raise CustomException(ex,sys)

class Customerdata:
    def __init__(self,
                 meanradius:float,meantexture:float,meansmoothness:float,meancompactness:float,meansymmetry:float,
                 meanfractaldimension:float,radiuserror:float,textureerror:float,smoothnesserror:float,compactnesserror:float,concavityerror:float,concavepointserror:float,symmetryerror:float,fractaldimensionerror:float,worstsmoothness:float,worstsymmetry:float,worstfractaldimension:float):
        self.meanradius=meanradius,
        self.meantexture=meantexture,
        self.meansmoothness=meansmoothness,
        self.meancompactness=meancompactness,
        self.meansymmetry=meansymmetry,
        self.meanfractaldimension=meanfractaldimension,
        self.radiuserror=radiuserror,
        self.textureerror=textureerror,
        self.smoothnesserror=smoothnesserror,
        self.compactnesserror=compactnesserror,
        self.concavityerror=concavityerror,
        self.concavepointserror=concavepointserror,
        self.symmetryerror=symmetryerror,
        self.fractaldimensionerror=fractaldimensionerror,
        self.worstsmoothness=worstsmoothness,
        self.worstsymmetry=worstsymmetry,
        self.worstfractaldimension=worstfractaldimension

    def get_dataas_dataframe(self):
        try:
            customer_input_data={
            'mean radius':self.meanradius,
            'mean texture':self.meantexture,
            'mean smoothness':self.meansmoothness,
            'mean compactness':self.meancompactness,
            'mean symmetry':self.meansymmetry,
            'mean fractal dimension':self.meanfractaldimension,
            'radius error':self.radiuserror,
            'texture error':self.textureerror,
            'smoothness error':self.smoothnesserror,
            'compactness error':self.compactnesserror,
            'concavity error':self.concavityerror,
            'concave points error':self.concavepointserror,
            'symmetry error':self.symmetryerror,
            'fractal dimension error':self.fractaldimensionerror,
            'worst smoothness':self.worstsmoothness,
            'worst symmetry':self.worstsymmetry,
            'worst fractal dimension':self.worstfractaldimension
            }

            df=pd.DataFrame(customer_input_data)
            print("Data frame values",df)
            return df
        except Exception as ex:
            raise CustomException(ex,sys)
                