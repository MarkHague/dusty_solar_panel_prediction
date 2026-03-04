import flask
from src.components.deploy_model import DeployModel

# load the model here before launching the server
deploy = DeployModel(
    model_save_path='../artifacts/training',
    model_name='model_mobnet_curated_data_extended'
)
app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict(threshold = 0.6):
    return deploy.predict(threshold=threshold)

if __name__ == "__main__":
    app.run()
