import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            df = pd.read_csv('some_data_set')
            logging.info('Read the dataset')

            os.makedirs(self.ingestion_config.train_data_path)

        except:
            pass