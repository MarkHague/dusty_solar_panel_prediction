import logging
import os.path

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer
from src.settings import LEARNING_RATE
import tensorflow as tf

from src.components.data_cleaning import DataCleaning


def train_model(learning_rate : int = LEARNING_RATE, epochs : int = 10,
                data_source : str = '../../solar_dust_detection/Detect_solar_dust_new_data',
                model_name : str = "model", **early_stopping_kwargs: any):

    # TODO add to docstring the steps taken in this function
    """
    Train the model to predict dusty/dirty and clean solar panels, using the following steps:
        1. Data cleaning (removing duplicates, image files not accepted by tensorflow etc.)
        2. Generate train, validation and test sets
        3. Add data augmentation layers and prefetch datasets
        4. Build the tf.keras.Model
        5. Compile and train the model with early stopping
        6. Save the model checkpoints using a callback

    Args:
        learning_rate: Learning rate for the Adam optimizer
        epochs: Number of training epochs
        data_source: Path to the data source directory
        model_name: Name for the saved model checkpoint
        **early_stopping_kwargs: Keyword arguments to pass to tf.keras.callbacks.EarlyStopping
            (e.g., monitor='val_loss', patience=3, min_delta=0.001)
    """
    logging.info("\n\n")
    logging.info("TRAINING MODEL..")
    # Data cleaning
    print("Data cleaning")
    data_cleaning = DataCleaning()
    data_cleaning.run_cleaning_steps(data_source=data_source)

    # Data Ingestion
    print("Data ingestion")
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
    print("Compiling model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    cp_callback = model_trainer.get_keras_cp_callback(model_name=model_name)

    # early stopping - use defaults if no kwargs provided
    early_stopping_params = {
        'monitor': 'val_loss',
        'min_delta': 0,
        'patience': 2,
        'verbose': 0,
        'baseline': None,
        'restore_best_weights': True,
        'start_from_epoch': 1,
        **early_stopping_kwargs # Override defaults with any provided kwargs
    }

    # noinspection PyArgumentList
    early_stop_callback = tf.keras.callbacks.EarlyStopping(**early_stopping_params)

    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=validation_ds,
                        callbacks=[early_stop_callback])

    model_save_path = os.path.join(model_trainer.model_trainer_config.trained_model_base_dir, model_name+".keras")
    os.makedirs(model_trainer.model_trainer_config.trained_model_base_dir, exist_ok=True)
    model.save(model_save_path)

    return model, history

if __name__ == "__main__":
    model_out, history_out = train_model(data_source='../../solar_dust_detection/Detect_solar_dust_curated',
                                 model_name="model_mobnet_curated_data_extended",
                                         patience = 3, verbose = 1)
