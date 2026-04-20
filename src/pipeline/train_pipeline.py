import logging
import os.path

import mlflow
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_builder import ModelBuilder
from src.settings import LEARNING_RATE, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VALIDATION_SPLIT, RAND_ROTATION, RAND_ZOOM
import keras
import json

from src.components.data_cleaning import DataCleaning


def train_model(learning_rate: int = LEARNING_RATE, epochs: int = 10,
                data_source: str = '../../solar_dust_detection/Detect_solar_dust_new_data',
                model_name: str = "model", run_name: str = None,
                base_model_class : keras.Model = keras.applications.MobileNetV2,
                **early_stopping_kwargs: any):

    """
    Train the model to predict dusty/dirty and clean solar panels, using the following steps:
        1. Data cleaning (removing duplicates, image files not accepted by tensorflow etc.)
        2. Generate train, validation and test sets
        3. Add data augmentation layers and prefetch datasets
        4. Build the keras.Model
        5. Compile and train the model with early stopping
        6. Save the model for deployment

    Args:
        learning_rate: Learning rate for the Adam optimizer
        epochs: Number of training epochs
        data_source: Path to the data source directory
        model_name: Name for the saved model checkpoint
        base_model_class: The keras model class. A list of available models can be found at https://keras.io/api/applications/
        run_name: Name of the run for tracking in mlflow
        **early_stopping_kwargs: Keyword arguments to pass to keras.callbacks.EarlyStopping
            (e.g., monitor='val_loss', patience=3, min_delta=0.001)
    """
    logging.info("\n\n")
    logging.info("TRAINING MODEL..")

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs requested": epochs,
            "batch_size": BATCH_SIZE,
            "img_height": IMG_HEIGHT,
            "img_width": IMG_WIDTH,
            "validation_split": VALIDATION_SPLIT,
            "rand_rotation": RAND_ROTATION,
            "rand_zoom": RAND_ZOOM,
            "model_name": model_name,
            "data_source": data_source,
            **{f"early_stopping_{k}": v for k, v in early_stopping_kwargs.items()},
        })

        # Data cleaning
        print("Data cleaning")
        data_cleaning = DataCleaning()
        data_cleaning.run_cleaning_steps(data_source=data_source)

        # Data Ingestion
        print("Data ingestion")
        data_ingestion = DataIngestion()
        train_ds, validation_ds, test_ds = data_ingestion.get_datasets(raw_data_path=data_source)
        # get the class names -> list
        class_names = train_ds.class_names

        # Log dataset sizes (approx: len() returns batch count, last batch may be partial)
        mlflow.log_params({
            "train_size": len(train_ds) * BATCH_SIZE,
            "val_size": len(validation_ds) * BATCH_SIZE,
            "test_size": len(test_ds) * BATCH_SIZE,
        })

        # Data Transformation
        data_transform = DataTransform()
        # Get Data Augmentation layers
        data_augmentation = data_transform.create_data_augmentation_layers()
        # Configure datasets for performance
        train_ds, validation_ds, test_ds = data_transform.prefetch_datasets(
            train_ds=train_ds, validation_ds=validation_ds, test_ds=test_ds)

        # build and train the model
        model_trainer = ModelBuilder()
        model = model_trainer.build_model(base_model_class= base_model_class,
                                         data_augmentation=data_augmentation)

        # Compile the model
        print("Compiling model")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

        # early stopping - use defaults if no kwargs provided
        early_stopping_params = {
            'monitor': 'val_loss',
            'min_delta': 0,
            'patience': 2,
            'verbose': 0,
            'baseline': None,
            'restore_best_weights': True,
            'start_from_epoch': 1,
            **early_stopping_kwargs  # Override defaults with any provided kwargs
        }

        # noinspection PyArgumentList
        early_stop_callback = keras.callbacks.EarlyStopping(**early_stopping_params)

        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=validation_ds,
                            callbacks=[early_stop_callback])

        # Log per-epoch metrics
        for metric, values in history.history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric, value, step=epoch)

        # Log final summary metrics for easy comparison in the runs table
        mlflow.log_metrics({
            "epochs_trained": len(history.history["loss"]),
            "best_val_loss": min(history.history["val_loss"]),
            "best_val_accuracy": history.history["val_accuracy"][np.argmin(history.history["val_loss"])],
        })

        model_save_path = os.path.join(model_trainer.model_trainer_config.trained_model_base_dir, model_name + ".keras")
        os.makedirs(model_trainer.model_trainer_config.trained_model_base_dir, exist_ok=True)

        model.save(model_save_path)
        # save the class names
        with open(os.path.join(model_trainer.model_trainer_config.trained_model_base_dir, model_name + "_class_names.json"), "w") as file:
            json.dump(sorted(class_names), file)

    return model, history
