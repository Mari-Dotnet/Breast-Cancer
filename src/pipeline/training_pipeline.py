import os,sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import Data_initiation

from src.components.data_transfermation import Data_tranfermation
from src.components.model_trainer import Data_model_initiation
from src.components.data_ingestion import Dataingestion_config

if __name__=="__main__":
    obj=Data_initiation()
    traindata_path,testdata_path=obj.initiate_dataingestion()
    print(f"traindata_path {traindata_path} testdatapath {testdata_path}")
    Data_tranfromation_obj=Data_tranfermation()
    train_arr,test_arr,processor_filepath=Data_tranfromation_obj.initiate_data_trianfromation(traindata_path,testdata_path)
    print('train_arr',train_arr.shape)
    print('test_arr',test_arr.shape)
    print('processor_filepath',processor_filepath)
    model_obj=Data_model_initiation()
    model_obj.Data_model_training(train_arr,test_arr) 

