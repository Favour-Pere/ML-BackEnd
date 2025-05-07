import joblib
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'heart_model.pkl')
# Load the model
model = joblib.load(file_path)

# Sample input — update with real values
sample = pd.DataFrame([{
    'Age': 60,
    'Sex': 1,
    'ChestPainType': 2,
    'RestingBP': 130,
    'Cholesterol': 250,
    'FastingBS': 0,
    'RestingECG': 1,
    'MaxHR': 140,
    'ExerciseAngina': 0,
    'Oldpeak': 1.0,
    'ST_Slope': 2
}])

prediction = model.predict(sample)
print("Prediction:", prediction[0])
