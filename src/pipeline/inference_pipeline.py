import io
from PIL import Image

from src.components.deploy_model import DeployModel


class InferencePipeline:
    """
    Orchestrates end-to-end inference: loads the model once at startup,
    then accepts raw image bytes and returns a prediction result.
    """
    def __init__(self, model_save_path: str, model_name: str):
        self.deploy_model = DeployModel(
            model_save_path=model_save_path,
            model_name=model_name
        )

    def run(self, image_bytes: bytes, threshold: float = 0.7) -> dict:
        """
        Runs the full inference pipeline on raw image bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            threshold: Decision boundary passed to the model predictor.

        Returns:
            dict with keys 'success' (bool) and 'predicted_label' (str).
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.deploy_model.predict(image=image, threshold=threshold)
