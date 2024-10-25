import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import streamlit as st
import pickle
import joblib

# Load the saved model, encoders, and scaler
model = pickle.load(open('churn_prediction_model.pkl', 'rb'))
scaler = joblib.load('standard_scaler.pkl')

st.title('Customer Churn Prediction')

st.sidebar.header('Input Customer Features')

# Collect input features from the sidebar
gender = st.sidebar.selectbox('Gender', ['female', 'male'])
SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
Partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
Dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
PhoneService = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
MultipleLines = st.sidebar.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
OnlineBackup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
StreamingTV = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.sidebar.number_input('Monthly Charges', min_value=0.0, value=50.0, step=10.0)
TotalCharges = st.sidebar.number_input('Total Charges', min_value=0.0, value=500.0, step=100.0)

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
def tenure_group(tenure):
    if tenure <= 12:
        return 0
    elif tenure <= 24:
        return 1
    elif tenure <= 48:
        return 2
    elif tenure <= 60:
        return 3
    else:
        return 4
input_df['TenureGroup'] = input_df['tenure'].apply(tenure_group)


# """
#
# Mapping for column 'TenureGroup':
# 0 -> 0-1 year
# 1 -> 1-2 years
# 2 -> 2-4 years
# 3 -> 4-5 years
# 4 -> 5+ years
# """

# Scale the numeric features using the previously saved scaler
print(input_df)


input_df = scaler.transform(input_df)

print(input_df)
input_data_array = input_df

# Make the prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_data_array)
    if prediction == 0:
        st.markdown('<h2 style="color: green; font-size: 24px;">This customer is not likely to churn.</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: red; font-size: 24px;">This customer is likely to churn.</h2>', unsafe_allow_html=True)
