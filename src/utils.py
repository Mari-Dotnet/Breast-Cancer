import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import accuracy_score

def Save_object(filepath:str,object):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        print("filepath",filepath)
        pickle.dump(object, open(filepath, 'wb'))
    except Exception as ex:
        raise CustomException(ex,sys)
    
def evaluatemetric(X_train,y_train,X_test,y_test,models):
    try:    
        result={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            #predict
            y_pred=model.predict(X_test)
            score=accuracy_score(y_test,y_pred)
            result[list(models.keys())[i]]=score
        return result
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(filepath):
    try:
        return pickle.load(open(filepath,'rb'))
    except Exception as e:
        raise CustomException(e,sys)


