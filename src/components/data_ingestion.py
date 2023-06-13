import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


from dataclasses import dataclass
@dataclass
class Dataingestion_config:
    train_datapth=os.path.join('artifacts','train.csv')
    test_datapth=os.path.join('artifacts','test.csv')
    raw_datapth=os.path.join('artifacts','raw.csv')


class Data_initiation:
    def __init__(self)-> None:
        self.data_ingestion=Dataingestion_config()

    def initiate_dataingestion(self):
        logging.info('Data ingestion started')
        try:
            data=load_breast_cancer()
            df=pd.DataFrame(data=data.data,columns=data.feature_names)
            df['Target']=data.target
            Corr_relation_Col=['mean perimeter', 'mean area', 'mean concavity',
       'mean concave points', 'perimeter error', 'area error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst compactness', 'worst concavity', 'worst concave points']
            df=df.drop(labels=Corr_relation_Col,axis=1)

            df.to_csv(self.data_ingestion.raw_datapth,header=True,index=False)
            logging.info('Raw data saved on folder')

            train_df,test_df=train_test_split(df,random_state=42,test_size=0.30)
            train_df.to_csv(self.data_ingestion.train_datapth,index=False,header=True)
            test_df.to_csv(self.data_ingestion.test_datapth,index=False,header=True)

            logging.info('train and test data saved on artifacts folder')

            return(
                self.data_ingestion.train_datapth,
                self.data_ingestion.test_datapth
            )
        except Exception as ex:
            logging.info("error occuerd on Data ingestion class")
            raise CustomException(ex,sys)


    

        
        