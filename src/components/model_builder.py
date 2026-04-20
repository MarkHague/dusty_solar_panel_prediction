import os
from dataclasses import dataclass
from src.settings import *
import keras

@dataclass
class ModelBuilderConfig:
    trained_model_base_dir: str ="../artifacts/training"

class ModelBuilder:
    def __init__(self):
        self.model_trainer_config=ModelBuilderConfig()
        self.PREPROCESS_FN = {
                    keras.applications.MobileNetV2: keras.applications.mobilenet_v2.preprocess_input,
                    keras.applications.NASNetMobile: keras.applications.nasnet.preprocess_input,
                }

    def build_model(self, base_model_class : keras.Model = keras.applications.MobileNetV2,
                    data_augmentation: keras.Sequential = None,
                    print_summary: bool = False) -> keras.Model:
        """
        Build the Mobilenet v2 architecture for transfer learning.

        Args:
            base_model_class: The keras model class. A list of available models can be found at https://keras.io/api/applications/
            train_ds: Training dataset
            data_augmentation: Pass in data augmentation layers
            print_summary: If True, print the model architecture summary
        Returns:
            model: The MobileNetV2 model with the base model frozen.
        """

        # Create the base model from the pre-trained model MobileNet V2
        base_model = base_model_class(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        # Freeze the convolutional base of MobileNet V2 - use this base as a feature extractor
        base_model.trainable = False

        # Add a classification head by averaging over the 8x8 spatial features
        global_average_layer = keras.layers.GlobalAveragePooling2D()

        # Add a final dense layer to produce a prediction for each image
        # this prediction will be treated as a logit.
        prediction_layer = keras.layers.Dense(1, activation='sigmoid')

        # chain together all the previous steps to create the final model
        # training = False because we have BatchNorm layers
        inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        if data_augmentation is not None:
            x = data_augmentation(inputs)
        else:
            x = inputs

        #check if preprocessing layer is builtin or not
        preprocess_fn = self.get_preprocessing_fn(base_model_class)

        if preprocess_fn is not None:
            preprocess_input = preprocess_fn
            x = preprocess_input(x)
        else:
            x = inputs
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = keras.Model(inputs, outputs)

        if print_summary:
            print(model.summary() )

        return model

    def get_preprocessing_fn(self, base_model_class: keras.Model):
        return self.PREPROCESS_FN.get(base_model_class, None)

    def get_keras_cp_callback(self, model_name: str = "model") -> keras.callbacks.ModelCheckpoint:
        """
        Creates a keras checkpoint callback to save model weights.

        Args:
            model_name: Name under which to save the model.
            base_save_dir: File path to where model is saved.
                           Combined with model_name to get unique dir for each model.
        """
        # create save dir
        save_dir = os.path.join(self.model_trainer_config.trained_model_base_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, model_name + ".weights.h5")

        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                         save_weights_only=True,
                                                         verbose=1)
        return cp_callback


