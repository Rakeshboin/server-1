"""
Flask server for crop recommendation predictions.

This server loads a RandomForestClassifier model using joblib and serves predictions via REST API.
The model expects 7 features: N, P, K, temperature, humidity, pH, and rainfall.

Usage:
    python server/app.py

Environment variables:
    MODEL_PATH: Path to the .joblib model file (default: random_forest_crop_recommendation_model.joblib)
    PORT: Server port (default: 5000)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import sys
import joblib  # <- Use joblib for loading scikit-learn models

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "random_forest_crop_recommendation_model.joblib")
PORT = int(os.environ.get("PORT", 5000))

CROP_LABELS = [
    "rice", "wheat", "maize", "cotton", "millets",
    "barley", "potato", "sugarcane", "coffee", "tea",
    "groundnut", "apple", "banana", "orange", "grapes",
    "papaya", "watermelon", "muskmelon", "mango", "pomegranate"
]

# Load model at startup
model = None
model_features = 7

def load_model():
    """Load the joblib model from file."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            return False

        print(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

if not load_model():
    print("WARNING: Model could not be loaded. Server will run but predictions will fail.")
    print("Make sure the model file exists and is accessible.")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "features_expected": model_features if model else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expected JSON request:
    {
      "features": {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 25.0,
        "humidity": 50.0,
        "ph": 6.7,
        "rainfall": 150.2
      }
    }

    Returns:
    {
      "prediction": "rice",
      "scores": [...],
      "confidence": ...
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs for details."}), 500

    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), 400

        features = data.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features' field in request"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse request: {e}"}), 400

    try:
        n = float(features.get("n", 0))
        p = float(features.get("p", 0))
        k = float(features.get("k", 0))
        temperature = float(features.get("temperature", 25.0))
        humidity = float(features.get("humidity", 50.0))
        ph = float(features.get("ph", 7.0))
        rainfall = float(features.get("rainfall", 100.0))

        if n < 0 or p < 0 or k < 0:
            return jsonify({"error": "N, P, K values must be non-negative"}), 400
        if ph < 0 or ph > 14:
            return jsonify({"error": "pH must be between 0 and 14"}), 400
        if rainfall < 0:
            return jsonify({"error": "Rainfall must be non-negative"}), 400

        x = np.array([[n, p, k, temperature, humidity, ph, rainfall]], dtype=np.float32)
        if x.shape[1] != model_features:
            return jsonify({
                "error": f"Expected {model_features} features, got {x.shape[1]}"
            }), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature values: {e}"}), 400

    try:
        # scikit-learn RandomForest: predict_proba gives probability for each class
        probabilities = model.predict_proba(x)[0]
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])
        prediction = CROP_LABELS[idx] if idx < len(CROP_LABELS) else f"crop_class_{idx}"

        return jsonify({
            "prediction": prediction,
            "scores": probabilities.tolist(),
            "confidence": confidence,
            "index": idx,
        })
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print(f"Starting FarmSense prediction server on port {PORT}...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print(f"Access the API at http://localhost:{PORT}/predict")
    print(f"Health check: http://localhost:{PORT}/health")
    app.run(host="0.0.0.0", port=PORT, debug=True)
