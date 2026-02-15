from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(_name_)
CORS(app)

# Load models
classification_model = joblib.load('classification_model.joblib')
regression_model = joblib.load('regression_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        required_features = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Moisture (%)']

        # Check missing fields
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Create dataframe
        features_df = pd.DataFrame([[data[f] for f in required_features]],
                                   columns=required_features)

        # Predictions
        predicted_fault_encoded = classification_model.predict(features_df)
        predicted_fault = label_encoder.inverse_transform(predicted_fault_encoded)[0]
        predicted_location = regression_model.predict(features_df)[0]

        return jsonify({
            "FaultType": predicted_fault,
            "FaultLocation": float(predicted_location)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if _name_ == "_main_":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
