import joblib
import streamlit as st

# Load the trained model
model = joblib.load('xgb_model.pkl')  # Ensure this file exists

def predict_loan_status(credit_policy, purpose, int_rate, installment, log_annual_inc, dti, fico):
    # Encode categorical variable
    purpose_mapping = {
        'debt_consolidation': 0, 
        'credit_card': 1, 
        'home_improvement': 2, 
        'other': 3
    }
    purpose_encoded = purpose_mapping[purpose]

    # Create input data
    input_data = [[credit_policy, purpose_encoded, int_rate, installment, log_annual_inc, dti, fico]]

    # Ensure the model is loaded before prediction
    prediction = model.predict(input_data)[0]  # Extract first element if it's an array
    return prediction

def loan_prediction_app():
    # Get user inputs for prediction
    credit_policy = st.slider('Credit Policy', 0, 1)
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
    int_rate = st.number_input('Interest Rate', min_value=0.0)
    installment = st.number_input('Installment', min_value=0.0)
    log_annual_inc = st.number_input('Log of Annual Income', min_value=0.0)
    dti = st.number_input('Debt to Income Ratio', min_value=0.0)
    fico = st.number_input('FICO Score', min_value=300, max_value=850)  # Correct range

    # Call prediction function
    prediction = predict_loan_status(credit_policy, purpose, int_rate, installment, log_annual_inc, dti, fico)
    
    # Display the prediction result
    st.write(f'Prediction: {prediction}')

# Run the app
if __name__ == '__main__':
    loan_prediction_app()
