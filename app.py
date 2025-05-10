from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('heart_disease_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = request.form['restecg']
        thalach = int(request.form['thalach'])
        exang = request.form['exang']
        oldpeak = float(request.form['oldpeak'])
        slope = request.form['slope']

        # Mapping the input values to encoded values
        sex_map = {'Male': 1, 'Female': 0}
        cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
        restecg_map = {'Normal': 1, 'ST-T abnormality': 2, 'Left ventricular hypertrophy': 0}
        exang_map = {'Yes': 1, 'No': 0}
        slope_map = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}

        # Create input sample for prediction
        sample = [
            age,
            sex_map.get(sex, 0),
            cp_map.get(cp, 0),
            trestbps,
            chol,
            fbs,
            restecg_map.get(restecg, 1),
            thalach,
            exang_map.get(exang, 0),
            oldpeak,
            slope_map.get(slope, 1)
        ]

        # Scaling input features
        sample_scaled = scaler.transform([sample])

        # Prediction
        prediction = model.predict(sample_scaled)
        prediction_prob = model.predict_proba(sample_scaled)[:, 1]

        # Output
        result = {
            "prediction": "Heart Disease" if prediction[0] == 1 else "No Heart Disease",
            "probability": f"{prediction_prob[0]*100:.2f}%",
            "risk_level": "High Risk ⚠️" if prediction_prob[0] > 0.8 else "Medium Risk ⚠️" if prediction_prob[0] > 0.5 else "Low Risk ✅",
            "suggestion": ""
        }

        # Add suggestion based on risk level
        if prediction[0] == 1:
            if prediction_prob[0] > 0.8:
                result["suggestion"] = "Immediate medical consultation is strongly recommended."
            elif prediction_prob[0] > 0.5:
                result["suggestion"] = "Please consult a doctor soon for further diagnosis."
            else:
                result["suggestion"] = "Consultation is advised to confirm diagnosis."
        else:
            result["suggestion"] = "Maintain a healthy lifestyle with regular checkups to stay safe."


        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
