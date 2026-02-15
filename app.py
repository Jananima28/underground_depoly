from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

from sklearn.metrics import accuracy_score

y_pred = classification_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Load ML models
classification_model = joblib.load('classification_model.joblib')
regression_model = joblib.load('regression_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')


# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Underground Cable Fault Prediction</title>
        </head>
        <body style="font-family: Arial; text-align:center; margin-top:50px;">
            <h2>Underground Cable Fault Prediction</h2>
            <form action="/predict" method="post">
                <input type="number" step="any" name="Voltage (V)" placeholder="Voltage (V)" required><br><br>
                <input type="number" step="any" name="Current (A)" placeholder="Current (A)" required><br><br>
                <input type="number" step="any" name="Temperature (C)" placeholder="Temperature (C)" required><br><br>
                <input type="number" step="any" name="Moisture (%)" placeholder="Moisture (%)" required><br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    '''


# ---------------- PREDICTION ROUTE ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept both JSON and Form Data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        required_features = [
            'Voltage (V)',
            'Current (A)',
            'Temperature (C)',
            'Moisture (%)'
        ]

        # Validate inputs
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input to DataFrame
        features_df = pd.DataFrame(
            [[float(data[f]) for f in required_features]],
            columns=required_features
        )

        # Make predictions
        predicted_fault_encoded = classification_model.predict(features_df)
        predicted_fault = label_encoder.inverse_transform(predicted_fault_encoded)[0]
        predicted_location = regression_model.predict(features_df)[0]

        # If JSON request → return JSON
        if request.is_json:
            return jsonify({
                "FaultType": predicted_fault,
                "FaultLocation": float(predicted_location)
            })

        # If browser form → return HTML result
        return f'''
        <html>
            <body style="font-family: Arial; text-align:center; margin-top:50px;">
                <h2>Prediction Result</h2>
                <p><b>Fault Type:</b> {predicted_fault}</p>
                <p><b>Fault Location:</b> {float(predicted_location)}</p>
                <br>
                <a href="/">Go Back</a>
            </body>
        </html>
        '''

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- MAIN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


