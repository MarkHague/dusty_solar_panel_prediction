import keras
from PIL import Image
from src.settings import *
import flask
from keras.preprocessing.image import img_to_array
import numpy as np
import io
import json
import os

class DeployModel:
    """
    Deploy a model using a Flask REST API
    """
    def __init__(self, model_save_path: str, model_name: str):
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.model = self.load_model()


    def load_model(self):
        """
        Loads a saved model from disk.
        """
        model_path = os.path.join(self.model_save_path, self.model_name+".keras")
        model = keras.models.load_model(model_path)
        return model

    def prepare_image(self, image=None, target_img_shape: tuple = (IMG_HEIGHT,IMG_WIDTH),
                      model_architecture='mobilnet_v2'):
        """
        Formats a user image (resizes, preprocesses) for passing to a model.

        Args:
            image: Input user image.
            target_img_shape: Shape used in training the model.
        """

        # if the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(target_img_shape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        if model_architecture == 'mobilenet_v2':
            image = keras.applications.mobilenet_v2.preprocess_input(image)

        # return the processed image
        return image

    def predict(self, threshold: float = 0.7):
        # initialize the data dictionary that will be returned from the
        # view
        data = {"success": False}

        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            if flask.request.files.get("image"):
                # read the image in PIL format
                image = flask.request.files["image"].read()
                image = Image.open(io.BytesIO(image))

                # preprocess the image and prepare it for classification
                image = self.prepare_image(image=image)

                pred = self.model.predict(image)
                # get the class names for the model we are using to predict
                with open(os.path.join(self.model_save_path, self.model_name + "_class_names.json"), 'r') as file:
                    class_names = json.load(file)

                    if pred <= threshold:
                        data["predicted_label"] = class_names[0]
                    else:
                        data["predicted_label"] = class_names[1]

                    # indicate that the request was a success
                    data["success"] = True

        # return the data dictionary as a JSON response
        return flask.jsonify(data)


