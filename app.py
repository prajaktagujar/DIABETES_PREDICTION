import os
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join("model", "svm_model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Error: Model or scaler file not found! Ensure they are inside the 'model/' folder.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form submission
        data = request.json
        required_features = ["pregnancies", "glucose", "bloodPressure", "skinThickness",
                             "insulin", "bmi", "dpf", "age"]

        # Ensure all required features are present
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert to numpy array and reshape
        input_data = np.array([[float(data[feature]) for feature in required_features]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        output = int(prediction[0])

        return jsonify({"prediction": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Ensure correct indentation for app startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get Railway-assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
