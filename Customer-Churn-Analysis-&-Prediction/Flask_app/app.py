from flask import Flask, render_template, request 
import pandas as pd
import pickle
import joblib


app = Flask(__name__)

# Load the saved model, encoders, and scaler
model = pickle.load(open('churn_prediction_model.pkl', 'rb'))
scaler = joblib.load('standard_scaler.pkl')

def tenure_group(tenure):
    if tenure <= 12:
        return 0
    elif tenure <= 24:
        return 1
    elif tenure <= 48:
        return 2
    elif tenure <= 60:
        return 
    else:
        return 4

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data from POST request
        gender = request.form['gender']
        SeniorCitizen = request.form['SeniorCitizen']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        tenure = int(request.form['tenure'])
        PhoneService = request.form['PhoneService']
        MultipleLines = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        StreamingTV = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        MonthlyCharges = float(request.form['MonthlyCharges'])
        TotalCharges = float(request.form['TotalCharges'])

        # Prepare input data
        input_data = {
            'gender': [1 if gender == 'male' else 0],
            'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
            'Partner': [1 if Partner == 'Yes' else 0],
            'Dependents': [1 if Dependents == 'Yes' else 0],
            'tenure': [tenure],
            'PhoneService': [1 if PhoneService == 'Yes' else 0],
            'MultipleLines': [2 if MultipleLines == 'Yes' else (1 if MultipleLines == 'No phone service' else 0)],
            'InternetService': [0 if InternetService == 'DSL' else (1 if InternetService == 'Fiber optic' else 2)],
            'OnlineSecurity': [2 if OnlineSecurity == 'Yes' else (1 if OnlineSecurity == 'No internet service' else 0)],
            'OnlineBackup': [2 if OnlineBackup == 'Yes' else (1 if OnlineBackup == 'No internet service' else 0)],
            'DeviceProtection': [2 if DeviceProtection == 'Yes' else (1 if DeviceProtection == 'No internet service' else 0)],
            'TechSupport': [2 if TechSupport == 'Yes' else (1 if TechSupport == 'No internet service' else 0)],
            'StreamingTV': [2 if StreamingTV == 'Yes' else (1 if StreamingTV == 'No internet service' else 0)],
            'StreamingMovies': [2 if StreamingMovies == 'Yes' else (1 if StreamingMovies == 'No internet service' else 0)],
            'Contract': [0 if Contract == 'Month-to-month' else (1 if Contract == 'One year' else 2)],
            'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0],
            'PaymentMethod': [2 if PaymentMethod == 'Electronic check' else (3 if PaymentMethod == 'Mailed check' else (0 if PaymentMethod == 'Bank transfer (automatic)' else 1))],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)
        input_df['TenureGroup'] = input_df['tenure'].apply(tenure_group)

        # Scale the numeric features using the saved scaler
        input_df = scaler.transform(input_df)

        # Make the prediction
        prediction = model.predict(input_df)

        # Render result page with prediction
        result_text = 'This customer is likely to churn.' if prediction[0] == 1 else 'This customer is not likely to churn.'
        return render_template('result.html', prediction=result_text)

    except KeyError as e:
        return f"Missing input: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)