from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Model Server is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Return the prediction
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
