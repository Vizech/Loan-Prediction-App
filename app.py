import joblib
import streamlit as st
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load('xgb_model.pkl')  # Ensure this file exists
training_columns = joblib.load('training_columns.pkl')  # Save this during training

def predict_loan_status(credit_policy, purpose, int_rate, installment, log_annual_inc, dti, fico):
    # Create a dictionary for the input data
    input_data = {
        'credit_policy': [credit_policy],
        'purpose': [purpose],
        'int_rate': [int_rate],
        'installment': [installment],
        'log_annual_inc': [log_annual_inc],
        'dti': [dti],
        'fico': [fico]
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # One-hot encode the 'purpose' column (same as during training)
    input_df = pd.get_dummies(input_df, columns=['purpose'], drop_first=True)

    # Ensure the input data has the same columns as the training data
    # Add missing columns with default value 0
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[training_columns]

    # Make prediction
    prediction = model.predict(input_df)[0]  # Extract first element if it's an array
    return prediction

def loan_prediction_app():
    st.title('Loan Repayment Prediction App')

    # Get user inputs for prediction
    credit_policy = st.slider('Credit Policy', 0, 1)
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
    int_rate = st.number_input('Interest Rate', min_value=0.0)
    installment = st.number_input('Installment', min_value=0.0)
    log_annual_inc = st.number_input('Log of Annual Income', min_value=0.0)
    dti = st.number_input('Debt to Income Ratio', min_value=0.0)
    fico = st.number_input('FICO Score', min_value=300, max_value=850)  # Correct range

    # Call prediction function
    if st.button('Predict'):
        prediction = predict_loan_status(credit_policy, purpose, int_rate, installment, log_annual_inc, dti, fico)
        
        # Display the prediction result
        if prediction == 0:
            st.success('Prediction: The loan will be fully paid.')
        else:
            st.error('Prediction: The loan will not be fully paid.')

# Run the app
if __name__ == '__main__':
    loan_prediction_app()