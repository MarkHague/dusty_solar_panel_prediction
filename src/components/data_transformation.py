from src.exception import CustomException
from src.logger import logging
import tensorflow as tf
from src.settings import *
from dataclasses import dataclass
import sys

@dataclass
class DataTransformConfig:
    rand_rotation: float=RAND_ROTATION
    rand_zoom: int=RAND_ZOOM


class DataTransform:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()

    def create_data_augmentation_layers(self):
        """
        Create data augmentation layers with random flip, rotation and zoom.
        """
        try:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(self.data_transform_config.rand_rotation),
                tf.keras.layers.RandomZoom(self.data_transform_config.rand_zoom)
            ])

            logging.info("Data augmentation layers created.")

            return data_augmentation

        except Exception as e:
            raise CustomException(e, sys)

    def prefetch_datasets(self, train_ds=None, validation_ds=None, test_ds=None):
        """
        Configure datasets for performance using prefetching (overlaps the preprocessing and model execution of a training step).
        """
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_ds.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_ds.prefetch(buffer_size=AUTOTUNE)

        return train_dataset, validation_dataset, test_dataset



    