import os,sys
import pandas as pd
import numpy as np
import pickle


from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.components.data_ingestion import Dataingestion_config

from src.utils import Save_object

from dataclasses import dataclass

@dataclass
class Data_tranformation_config:
    pre_processor_filepath=os.path.join('artifacts','preprocessor.pkl')

class Data_tranfermation:
    def __init__(self):
        self.preprocessor_config=Data_tranformation_config()

    def get_data_transfermation_obj(self):
        try:
            logging.info("get data transformation started")
            df=pd.read_csv(Dataingestion_config.raw_datapth)
            df=df.drop(labels='Target',axis=1)
            Numeric_column=df.columns
            Numeric_pipeline=Pipeline(
                steps=[
                ('impute',SimpleImputer(strategy='median')),
                ('sclaer',StandardScaler())

                ]
            )
            logging.info('numeric pipeline initialized')
            preprocessor=ColumnTransformer([
                ('Numeric_pipeline',Numeric_pipeline,Numeric_column)
            ]
            )
            logging.info('prepocessor initialized')

            return preprocessor
        except Exception as ex:
            raise CustomException(ex,sys)
            

            
        except Exception as ex:
            raise CustomException(ex,sys)
    def initiate_data_trianfromation(self,train_path,test_path):
        try:
            logging.info("Initiate Data tranfromation started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f"train and test data loaded  - trian data:{train_df.shape}- -  test_df:{test_df.shape}")
            
            target='Target'
            #Split input and output feature trian
            input_feature_trin_df=train_df.drop(labels='Target',axis=1)
            input_feature_trin_target=train_df['Target']

            #Split input and output feature test
            input_feature_test_df=test_df.drop(labels='Target',axis=1)
            input_feature_test_target=test_df['Target']

            logging.info('pre processing started')
            preprocessor_obj=self.get_data_transfermation_obj()
            
            
            input_feature_trin_df=preprocessor_obj.fit_transform(input_feature_trin_df)
            input_feature_test_df=preprocessor_obj.transform(input_feature_test_df)

           
            logging.info("Prrpcoessor completed")

            train_arr=np.c_[input_feature_trin_df,input_feature_trin_target]
            test_arr=np.c_[input_feature_test_df,input_feature_test_target]

            print("processor path",self.preprocessor_config.pre_processor_filepath)
            # save the preprocessing object
            # Save_object(
            #     filepath=self.preprocessor_config.pre_processor_filepath,
            #     obj=preprocessor_obj
            # )
            pickle.dump(preprocessor_obj, open(self.preprocessor_config.pre_processor_filepath, 'wb'))

            return(
                train_arr,
                test_arr,
                self.preprocessor_config.pre_processor_filepath

            )

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
    except Exception as ex:
        raise CustomException(ex,sys)