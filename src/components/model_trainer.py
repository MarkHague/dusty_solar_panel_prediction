import os
import sys
from dataclasses import dataclass
from src.settings import *
import tensorflow as tf

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def build_mobilenet_v2_model(self, train_ds = None, data_augmentation: tf.keras.Sequential = None,
                                 print_summary: bool = False) -> tf.keras.Model:
        """
        Build the Mobilenet v2 architecture for transfer learning.

        Args:
            train_ds: Training dataset
            data_augmentation:
            print_summary: If True, print the model architecture summary
        Returns:
            model: The Mobilenet v2 model with
        """
        # layer that will rescale input images to [-1, 1]
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        # take a look at the output shape of the feature batch
        image_batch, label_batch = next(iter(train_ds))
        feature_batch = base_model(image_batch)
        # Freeze the convolutional base of MobileNet V2 - use this base as a feature extractor
        base_model.trainable = False

        # Add a classification head by averaging over the 8x8 spatial features
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)

        # Add a final dense layer to produce a prediction for each image
        # this prediction will be treated as a logit. Positive numbers predict class 1, negative numbers predict class 0.
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        prediction_batch = prediction_layer(feature_batch_average)

        # chain together all the previous steps to create the final model
        # training = False because we have BatchNorm layers
        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        if data_augmentation is not None:
            x = data_augmentation(inputs)
        else:
            x = inputs
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        if print_summary:
            print(model.summary() )

        return model


