from src.pipeline.train_pipeline import train_model

if __name__ == "__main__":
    model_out, history_out = train_model(data_source='../../solar_dust_detection/Detect_solar_dust_new_data',
                                         model_name="model_mobnet_curated_data_extended",
                                         patience=3, verbose=1)
