from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Data:", data)  # Debugging

        # Convert input data to array
        features = np.array([
            data["age"], data["sex"], data["cp"],
            data["trestbps"], data["chol"], data["restecg"]
        ]).reshape(1, -1)

        print("Transformed Features (Before Scaling):", features)  # Debugging

        # Scale the features
        features = scaler.transform(features)

        print("Transformed Features (After Scaling):", features)  # Debugging

        # Make prediction
        prediction = model.predict(features)[0]

        print("Model Prediction:", prediction)  # Debugging

        return jsonify({"prediction": int(prediction)})  # 0 = Healthy, 1 = Prone to Heart Disease

    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
