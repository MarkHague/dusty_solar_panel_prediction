import keras
from PIL import Image
from src.settings import *
from keras.preprocessing.image import img_to_array
import numpy as np
import json
import os


class DeployModel:
    """
    Loads a trained model and runs inference on preprocessed images.
    """
    def __init__(self, model_save_path: str, model_name: str):
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        """
        Loads a saved model from disk.
        """
        model_path = os.path.join(self.model_save_path, self.model_name + ".keras")
        model = keras.models.load_model(model_path)
        return model

    def prepare_image(self, image: Image.Image, target_img_shape: tuple = (IMG_HEIGHT, IMG_WIDTH)):
        """
        Formats a PIL image (resizes, converts) for passing to the model.
        Note: pixel scaling is handled by the preprocessing layer inside the model.

        Args:
            image: Input PIL image.
            target_img_shape: Shape used in training the model.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(target_img_shape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image: Image.Image, threshold: float = 0.7) -> dict:
        """
        Runs inference on a PIL image and returns a result dict.

        Args:
            image: PIL image to classify.
            threshold: Decision boundary between class 0 and class 1.

        Returns:
            dict with keys 'success' (bool) and 'predicted_label' (str).
        """
        prepared = self.prepare_image(image=image)
        pred = self.model.predict(prepared)

        with open(os.path.join(self.model_save_path, self.model_name + "_class_names.json"), 'r') as file:
            class_names = json.load(file)

        predicted_label = class_names[1] if pred > threshold else class_names[0]
        return {"success": True, "predicted_label": predicted_label}


