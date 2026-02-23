import os
import boto3
from dotenv import load_dotenv
import tensorflow as tf
from src.logger import logging
from src.settings import *
from src.exception import CustomException
import numpy as np
import sys
import pandas as pd

def get_s3_client() -> None:

    """
    Get an AWS S3 client to retrieve training data.
    """
    load_dotenv()

    return boto3.client(
        "s3",
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')

    )

def get_model_predictions(model_path: str = None, image_dataset_path: str = None) -> pd.DataFrame:
    """
    Compare model predictions to expected labels.

    Args:
        model_path: File path to the .keras saved model.
        image_dataset_path: File path to the newly extracted images.
    Returns:
        A pandas Dataframe containing the model predictions (raw probabilities, and labels), and the expected labels.
    """

    logging.info("\n")
    logging.info(f"Creating model predictions using model {model_path}...")
    model = tf.keras.models.load_model(model_path)

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dataset_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Keep order so filenames line up with predictions
        labels='inferred',  # No labels — we just want to run inference
        label_mode='binary',
    )

    class_names = dataset.class_names
    # Get file paths so you can match predictions back to images
    file_paths = dataset.file_paths

    # get the expected labels - i.e. the label we expect based on the search term
    expected_labels = np.concatenate([y.numpy() for _, y in dataset])

    # Generate predictions
    try:
        predictions = model.predict(dataset)

        df = pd.DataFrame(columns=['image_filepath', 'model_prob', 'predicted_label', 'expected_label'])
        for path, expected, pred in zip(file_paths, expected_labels, predictions):
            prob = float(pred[0])
            predicted_label = class_names[1] if prob >= 0.5 else class_names[0]
            expected_label = class_names[int(expected[0])]

            new_row = {
                "image_filepath": path,
                "model_prob": prob,
                "predicted_label": predicted_label,
                "expected_label": expected_label
            }

            df.loc[len(df)] = new_row

        return df

    except Exception as e:
        raise CustomException(e, sys)

