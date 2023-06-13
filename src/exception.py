import sys
from src.logger import logging


def eror_message_details(error_message,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in pyhton script name {0} line number {1} error message {2}".format(
      file_name,exc_tb.tb_lineno,str(error_message))
    return error_message

class CustomException:
    def __init__(self,error_message,error_detail:sys):
        super.__init__(error_message)
        self.error_msg=eror_message_details(error_message,error_detail)
    
    def __str__(self) -> str:
        return self.error_msg
    

if __name__=="__main__":
    try:
        logging.info('Admin logged in')
        print("method inside")
        logging.info("Logging started")
        a=12/0
        pass
    except Exception as e:
        logging.info("error occured")
        raise CustomException(e,sys)