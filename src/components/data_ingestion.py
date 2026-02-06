import os
import sys
from src.logger import logging
from src.exception import CustomException
import tensorflow as tf
from src.settings import *

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # train_data_path: str=os.path.join('artifacts',"train.csv")
    # test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str="../../../solar_dust_detection/Detect_solar_dust_new_data"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=self.ingestion_config.raw_data_path,
                validation_split=VALIDATION_SPLIT,
                subset='training',
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE
            )

            validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=self.ingestion_config.raw_data_path,
                validation_split=VALIDATION_SPLIT,
                subset='validation',
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE
            )

            # generate test set
            val_batches = tf.data.experimental.cardinality(validation_ds)
            test_ds = validation_ds.take(val_batches // 3)
            validation_ds = validation_ds.skip(val_batches // 3)

            logging.info('Train, validation and test splits completed')
            logging.info('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_ds))
            logging.info('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

            return(
                train_ds,
                validation_ds,
                test_ds
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()