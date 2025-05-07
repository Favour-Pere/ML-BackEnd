import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'heart_model.pkl')
app = Flask(__name__)

model = joblib.load(file_path)

expected_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not all(feature in data for feature in expected_features):
        return jsonify({"error": "Missing features in the input data"}), 400
    
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'

    return jsonify({"prediction": int(prediction), 'result' : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
