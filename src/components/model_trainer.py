import os,sys
import pandas as pd
import numpy as np
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from src.components.data_ingestion import Dataingestion_config

from src.components.data_transfermation import Data_tranfermation

from src.utils import evaluatemetric
from src.utils import Save_object

from dataclasses import dataclass
@dataclass
class Datamodel_config:
    Datamodel_filepath=os.path.join('artifacts','model_processor.pkl')

class Data_model_initiation:
    def __init__(self)-> None:
        self.Data_modelpath=Datamodel_config()
    
    def Data_model_training(self,train_arr,test_arr):
        try:
            logging.info("Modle training started")
            
            X_train,y_train,X_test,y_test=(
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:,:-1],
                    test_arr[:,-1]
            )
             
            
            models={
                "LogisticRegression":LogisticRegression(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "SVC":SVC(),
                "RandomForestClassifier":RandomForestClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "GaussianNB":GaussianNB()
            }

            result:dict=evaluatemetric(X_train,y_train,X_test,y_test,models)
            print("values",result.values())
            print("keys",result.keys())
            print("result",result)
           
            best_score=max(result.values())
            print("best_score",best_score)
            best_model=list(result.keys())[
                list(result.values()).index(best_score)
            ]
            print("best_model",best_model)
            
            logging.info(f"best model as {best_model}  and the score is  {best_score}")

            print(f"best model as {best_model}  and the score is  {best_score}")

            best_model_obj=models[best_model]

            # Save_object(
            #     filepath=self.Data_modelpath.Datamodel_filepath,
            #     obj=best_model_obj
            # )
            pickle.dump(best_model_obj, open(self.Data_modelpath.Datamodel_filepath ,'wb'))

            logging.info("best model identified")

        except Exception as ex:
            raise CustomException(ex,sys)


if __name__=="__main__":
    try:
        logging.info("data tranfermation started")
        train_path=Dataingestion_config.train_datapth
        test_path=Dataingestion_config.test_datapth

        obj=Data_tranfermation()
        train_arr,test_arr,processor_filepath=obj.initiate_data_trianfromation(train_path,test_path)
        
        print(train_arr.shape,test_arr.shape)

        datamodl_obj=Data_model_initiation()
        datamodl_obj.Data_model_training(train_arr,test_arr)
        
    except Exception as ex:
        raise CustomException(ex,sys)

