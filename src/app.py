import os
import flask
from src.pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model_save_path=os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'training')),
    model_name='model_mobnet_curated_data_extended'
)

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not flask.request.files.get("image"):
        return flask.jsonify({"success": False}), 400

    image_bytes = flask.request.files["image"].read()
    result = pipeline.run(image_bytes=image_bytes)
    return flask.jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
