import logging
import os
from datetime import datetime

time_as_object = datetime.today()

time_as_string = time_as_object.strftime('%Y-%m-%d')
                                         
LogFile=f"{time_as_string}.log"
logs_path=os.path.join(os.getcwd(),'logs',LogFile)
os.makedirs(logs_path,exist_ok=True)

Log_filepath=os.path.join(logs_path,LogFile)

logging.basicConfig(

    filename=Log_filepath,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)