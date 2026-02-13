from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer
from src.settings import LEARNING_RATE
import tensorflow as tf

from src.components.data_cleaning import DataCleaning
import glob

from tests.test_data_cleaning import data_cleaning


def train_model(learning_rate = LEARNING_RATE, epochs = 10,
                data_source = '../../solar_dust_detection/Detect_solar_dust_new_data'):
    """
    Train the model to predict dusty/dirty and clean solar panels.

    """
    # Data cleaning
    data_cleaning = DataCleaning()
    data_cleaning.run_cleaning_steps(data_source=data_source)

    # Data Ingestion
    data_ingestion = DataIngestion()
    train_ds, validation_ds, test_ds = data_ingestion.get_datasets(raw_data_path=data_source)


    # Data Transformation
    data_transform = DataTransform()
    # Get Data Augmentation layers
    data_augmentation = data_transform.create_data_augmentation_layers()
    # Configure datasets for performance
    train_ds, validation_ds, test_ds = data_transform.prefetch_datasets(
        train_ds=train_ds, validation_ds=validation_ds, test_ds=test_ds)

    # build and train the model
    model_trainer = ModelTrainer()
    model = model_trainer.build_mobilenet_v2_model(train_ds=train_ds, data_augmentation=data_augmentation)

    # Compile the model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=validation_ds)

    return model, history

