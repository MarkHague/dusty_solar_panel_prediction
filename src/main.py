import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv
from src.pipeline.train_pipeline import train_model

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
default_tracking_uri = "sqlite:///" + str(PROJECT_ROOT / "artifacts" / "mlflow.db")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri))

if __name__ == "__main__":
    mlflow.set_experiment("mobnet_curated_data_extended_more_epochs")
    model_out, history_out = train_model(data_source='../../solar_dust_detection/Detect_solar_dust_new_data',
                                         model_name="model_mobnet_curated_data_extended",
                                         run_name="image_size_512",
                                         patience=8, verbose=1, epochs=25)
