import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__, static_url_path='/Flask/static')

# Load saved model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Details page
@app.route('/details')
def details():
    return render_template('details.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve values from form
    Gender = float(request.form['Gender'])
    Age = float(request.form['Age'])
    Patient = float(request.form['Patient'])
    Severity = float(request.form['Severity'])
    BreathShortness = float(request.form['BreathShortness'])
    VisualChange = float(request.form['VisualChange'])
    NoseBleeding = float(request.form['NoseBleeding'])
    WhenDiagnosed = float(request.form['WhenDiagnosed'])
    Systolic = float(request.form['Systolic'])
    Diastolic = float(request.form['Diastolic'])
    ControlledDiet = float(request.form['ControlledDiet'])

    # Prepare data for prediction
    features_values = np.array([[
        Gender, Age, Patient, Severity, BreathShortness,
        VisualChange, NoseBleeding, WhenDiagnosed,
        Systolic, Diastolic, ControlledDiet
    ]])

    df = pd.DataFrame(features_values, columns=[
        'Gender', 'Age', 'Patient', 'Severity', 'BreathShortness',
        'VisualChange', 'NoseBleeding', 'WhenDiagnosed',
        'Systolic', 'Diastolic', 'ControlledDiet'
    ])

    # Model prediction
    prediction = model.predict(df)


    # Interpretation
    if prediction[0] == 0:
        result = "NORMAL"
    elif prediction[0] == 1:
        result = "HYPERTENSION (Stage-1)"
    elif prediction[0] == 2:
        result = "HYPERTENSION (Stage-2)"
    else:
        result = "HYPERTENSIVE CRISIS"

    return render_template('prediction.html', prediction_text=f"Your Blood Pressure stage is: {result}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
