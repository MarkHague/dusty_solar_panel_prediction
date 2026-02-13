import sys
from src.logger import logging
from src.exception import CustomException
import tensorflow as tf
from src.settings import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    validation_split: float=VALIDATION_SPLIT
    batch_size: int=BATCH_SIZE
    image_height: int=IMG_HEIGHT
    image_width: int = IMG_WIDTH

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_datasets(self, raw_data_path: str):
        """
        Return train, validation and test datasets using keras image_dataset_from_directory method.
        """
        logging.info("\n")
        logging.info("Entered data ingestion method")
        try:
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=raw_data_path,
                validation_split=self.ingestion_config.validation_split,
                subset='training',
                seed=123,
                image_size=(self.ingestion_config.image_height, self.ingestion_config.image_width),
                batch_size= self.ingestion_config.batch_size
            )

            validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=raw_data_path,
                validation_split=self.ingestion_config.validation_split,
                subset='validation',
                seed=123,
                image_size=(self.ingestion_config.image_height, self.ingestion_config.image_width),
                batch_size= self.ingestion_config.batch_size
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