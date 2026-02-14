from flask import Flask, request, jsonify, Flask
import pandas as pd
import joblib

app = Flask(__name__)

# Load the classification model
classification_model = joblib.load('/content/classification_model.joblib')
print("Classification model loaded successfully.")

# Load the regression model
regression_model = joblib.load('/content/regression_model.joblib')
print("Regression model loaded successfully.")

# Load the LabelEncoder
label_encoder = joblib.load('/content/label_encoder.joblib')
print("LabelEncoder loaded successfully.")


# Initialize the Flask application (if not already done)
# app = Flask(__name__)

# Assuming classification_model, regression_model, and label_encoder are loaded as in previous steps

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    required_features = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Moisture (%)']
    input_features = {}
    for feature in required_features:
        if feature not in data:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
        input_features[feature] = data[feature]

    try:
        # Create a DataFrame from the input features
        features_df = pd.DataFrame([input_features])

        # Predict FaultType (encoded label)
        predicted_fault_type_encoded = classification_model.predict(features_df)

        # Inverse transform to get original FaultType string
        predicted_fault_type = label_encoder.inverse_transform(predicted_fault_type_encoded)[0]

        # Predict FaultLocation
        predicted_fault_location = regression_model.predict(features_df)[0]

        # Construct the response dictionary
        result = {
            'FaultType': predicted_fault_type,
            'FaultLocation': predicted_fault_location
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("Prediction logic implemented in the '/predict' route.")

if __name__ == "__main__":
    app.run(debug=True)
