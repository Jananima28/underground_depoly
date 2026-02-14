from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models (from project folder)
classification_model = joblib.load('classification_model.joblib')
regression_model = joblib.load('regression_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Send POST request with JSON data"
    
    data = request.get_json(force=True)

    required_features = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Moisture (%)']
    
    features_df = pd.DataFrame([[data[f] for f in required_features]],
                               columns=required_features)

    predicted_fault_encoded = classification_model.predict(features_df)
    predicted_fault = label_encoder.inverse_transform(predicted_fault_encoded)[0]
    predicted_location = regression_model.predict(features_df)[0]

    return jsonify({
        "FaultType": predicted_fault,
        "FaultLocation": float(predicted_location)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
