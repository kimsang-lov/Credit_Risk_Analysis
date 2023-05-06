import streamlit as st
import pandas as pd
import joblib
from sklearn.base import TransformerMixin, BaseEstimator

# Load the trained machine learning model
# attrib_adder = joblib.load('/home/kafka/Documents/CreditRiskAnalysis/attrib_adder.joblib')
model = joblib.load('/home/kafka/Documents/CreditRiskAnalysis/XGB_Pipeline.joblib')

# Define the features used by the model
features = ['Age', 'Annual Income', 'Home Status', 'Employment Length', 'Loan Intent',
            'Loan Amount', 'Loan Grade', 'Interest Rate', 'Loan to Income Ratio',
            'Historical Default', 'Credit History Length']

# Create a function to get user input
def get_user_input():
    person_age = st.sidebar.slider('Age', 18, 100, 25)
    person_income = st.sidebar.slider('Annual Income', 0, 2000000, 50000)
    person_home_ownership = st.sidebar.selectbox('Home Status', ['RENT', 'MORTGAGE', 'OWN', "OTHER"])
    person_emp_length = st.sidebar.slider('Employment Length', 0, 100, 5)
    loan_intent = st.sidebar.selectbox('Loan Intent', ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION' 'HOMEIMPROVEMENT',])
    loan_amnt = st.sidebar.slider('Loan Amount', 100, 50000, 1000)
    loan_grade = st.sidebar.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_int_rate = st.sidebar.slider('Interest Rate', 1, 24, 10)
    # loan_percent_income = st.sidebar.slider('Loan to Income Ratio', 0.0, 1.0, 0.5)
    cb_person_default_on_file = st.sidebar.selectbox('Historical Default', ['Y', 'N'])
    cb_person_cred_hist_length = st.sidebar.slider('Credit History Length', 0, 50, 5)

    # Create a dictionary with the user inputs
    user_data = {'Age': person_age,
                 'Annual Income': person_income,
                 'Home Status': person_home_ownership,
                 'Employment Length': person_emp_length,
                 'Loan Intent': loan_intent,
                 'Loan Grade': loan_grade,
                 'Loan Amount': loan_amnt,
                 'Interest Rate': loan_int_rate,
                 'Loan to Income Ratio': loan_amnt / person_income,
                 'Historical Default': cb_person_default_on_file,
                 'Credit History Length': cb_person_cred_hist_length}

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

    data = user_input.copy()
    data.columns = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length']

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_total_payment=True):
            self.add_total_payment = add_total_payment
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            int_rate_range = pd.Series(pd.cut(X['loan_int_rate'],
                                              bins=[0, 14, 24], labels=['low', 'high']), name='int_rate_range')
            loan_to_income_range = pd.Series(pd.cut(X['loan_percent_income'],
                                                    bins=[0, 0.2, 0.5, 1.], labels=['low', 'medium', 'high']), name='loan_to_income_range')

            if self.add_total_payment:
                total_payment = pd.Series(X['loan_amnt'] + X['loan_amnt'] * (X['loan_int_rate'] / 100), name='total_payment')
                return pd.concat([X, int_rate_range, loan_to_income_range,total_payment], axis=1)
            return pd.concat([X, int_rate_range, loan_to_income_range], axis=1)


    attr_adder = CombinedAttributesAdder()
    user_input = attr_adder.fit_transform(data)

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
