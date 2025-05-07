import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'heart.csv')

df = pd.read_csv(file_path)

df_encoded = df.copy()

categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

encoder = LabelEncoder()
for column in categorical_columns:
    df_encoded[column] = encoder.fit_transform(df[column])


x = df_encoded.drop('HeartDisease', axis=1)

y = df_encoded['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report\n", classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(script_dir, 'heart_model.pkl'))