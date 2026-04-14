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
    mlflow.set_experiment("mobnet_curated_data_only")
    model_out, history_out = train_model(data_source='../../solar_dust_detection/Detect_solar_dust_curated',
                                         model_name="model_mobnet_curated_data",
                                         patience=3, verbose=1)
