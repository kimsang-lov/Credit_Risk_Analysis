import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load('XGB_Pipeline.joblib')

# Define the features used by the model
features = ['Age', 'Annual Income', 'Home Status', 'Employment Length', 'Loan Intent',
            'Loan Amount', 'Loan Grade', 'Interest Rate', 'Loan to Income Ratio',
            'Historical Default', 'Credit History Length']

# Create a function to get user input
def get_user_input():
    age = st.sidebar.slider('Age', 18, 100, 25)
    annual_income = st.sidebar.slider('Annual Income', 0, 500000, 50000)
    home_status = st.sidebar.selectbox('Home Status', ['Rent', 'Mortgage', 'Own'])
    employment_length = st.sidebar.slider('Employment Length', 0, 50, 5)
    loan_intent = st.sidebar.selectbox('Loan Intent', ['Education', 'Medical', 'Venture', 'Home Improvement', 'Personal', 'Debt Consolidation'])
    loan_amount = st.sidebar.slider('Loan Amount', 1000, 100000, 10000)
    loan_grade = st.sidebar.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    interest_rate = st.sidebar.slider('Interest Rate', 1, 30, 10)
    loan_to_income_ratio = st.sidebar.slider('Loan to Income Ratio', 0, 1, 0.5)
    historical_default = st.sidebar.selectbox('Historical Default', ['Y', 'N'])
    credit_history_length = st.sidebar.slider('Credit History Length', 0, 50, 5)

    # Create a dictionary with the user inputs
    user_data = {'Age': age,
                 'Annual Income': annual_income,
                 'Home Status': home_status,
                 'Employment Length': employment_length,
                 'Loan Intent': loan_intent,
                 'Loan Amount': loan_amount,
                 'Loan Grade': loan_grade,
                 'Interest Rate': interest_rate,
                 'Loan to Income Ratio': loan_to_income_ratio,
                 'Historical Default': historical_default,
                 'Credit History Length': credit_history_length}

    # Convert the dictionary into a Pandas dataframe
    user_df = pd.DataFrame(user_data, index=[0])

    return user_df

# Create a function to make predictions using the model
def predict_loan_default(user_input):
    prediction = model.predict(user_input)
    return prediction

# Create the Streamlit app
def app():
    st.title('Credit Risk Analysis')
    st.sidebar.header('User Input Features')

    # Get user input
    user_input = get_user_input()

    # Display the user input
    st.header('User Input Features')
    st.write(user_input)

    # Make predictions
    prediction = predict_loan_default(user_input)

    # Display the prediction
    st.header('Prediction')
    if prediction[0] == 1:
        st.write('This loan is at risk of defaulting.')
    else:
        st.write('This loan is not at risk of defaulting.')

if __name__ == '__main__':
    app()