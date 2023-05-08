import streamlit as st
import pandas as pd
import joblib
from sklearn.base import TransformerMixin, BaseEstimator

# Load the trained machine learning model
# attrib_adder = joblib.load('/home/kafka/Documents/CreditRiskAnalysis/attrib_adder.joblib')

model = joblib.load('XGB_Pipeline.joblib')

# Define the features used by the model
features = ['Age', 'Annual Income', 'Home Status', 'Employment Length', 'Loan Intent',
            'Loan Amount', 'Loan Grade', 'Interest Rate', 'Loan to Income Ratio',
            'Historical Default', 'Credit History Length']

# Create a function to get user input
def get_user_input():
    person_age = st.sidebar.slider('Age', 18, 100, 25)
    person_income = st.sidebar.slider('Annual Income', 0, 200000, 50000)
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
        st.markdown('#### :red[This loan is at risk of defaulting.]')
    else:
        st.markdown('#### :green[This loan is not at risk of defaulting.]')

if __name__ == '__main__':
    app()

import pandas as pd
import numpy as np
from PIL import Image

st.subheader('Documentation')
st.write('''
Credit risk analysis is a critical task for financial institutions, as it helps to minimize the risk associated with 
lending money. Accurately predicting whether a borrower is likely to default on loan payments is key to maintaining a 
healthy portfolio and minimizing losses. In this project, we aim to develop a machine learning model to predict the 
likelihood of default for borrowers using various data science techniques.
''')
st.code('''
    # Import necessary library
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    %matplotlib inline

    data = pd.read_csv("Datasets/credit_risk_dataset.csv")
    data.head()''')

data = pd.read_csv('Datasets/credit_risk_dataset.csv')
st.write(data.head())

st.markdown('''
### Dataset
This dataset contains data for :green[___32,581___] borrowers and :green[___12___] features related to each of them.
- :blue[Age]: numerical variables, age in years
- :blue[Annual Income]: numerical variable; annual income in dollars
- :blue[Home Status]: categorical variable; “rent”, “mortgage”, “own”
- :blue[Employment length]: numerical variable; employment length in years
- :blue[Loan intent]: categorical variable; “education”, “medical”, “venture”, “home improvement”, “personal”, or “debt consolidation”
- :blue[Loan amount]: numerical variable; loan amount in dollars
- :blue[Loan grade]: categorical variable; “A”, “B”, “C”, “D”, “E”, “F”, or “G”
- :blue[Interest rate]: numerical variable; interest rate in percentage
- :blue[Loan to income ratio]: numerical variable; between 0 and 1
- :blue[Historical default]: binary categorical variable; “Y” or “N”
- :blue[Loan status]: binary numerical variable; 0 (no defualt) or 1 (default). This is going to be our target variable.
- :blue[Credit History Lenght]: numerical variable; credit lenght in years
''')

st.write('### Data Exploration and Preprocessing')

st.code('''
# Check the data types of each column
print(data.dtypes)
''')
# Check the data types of each column
st.write(data.dtypes)

st.code('''
# Calculate number of missing values in each column
missing_values = data.isna().sum().sort_values(ascending=False)

# Calculate percentage of missing values in each column
percent_missing = (missing_values / len(data)) * 100

# Combine number and percentage of missing values into a dataframe
missing_data = pd.concat([missing_values, percent_missing], axis=1, keys=['Total', 'Percent'])

# Sort dataframe by percentage of missing values
missing_data = missing_data.sort_values('Percent', ascending=False)

''')
img0 = Image.open('images/img0.png')
st.image(img0)
st.write('''
We can see that `Employment length` and `Interest rate` both have missing values. Given that the missing values 
represent a small percentage of the dataset, we'll remove the rows that contain missing values.
''')

st.code('''
# Drop missing values
data.dropna(axis=0, inplace=True)

# Check the summary statistics of the numerical variables
data.describe()
''')
# Drop missing values
data.dropna(axis=0, inplace=True)
st.write(data.describe())

#
st.code('''
# Check the distribution of numerical varaibles
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
for i, col in enumerate(num_cols):
    sns.boxplot(ax=axes[i//3, i%3], data=data, y=col)
    
for i in [1, 2]:
    fig.delaxes(axes[2][i])
''')

image2 = Image.open('images/img2.png')
st.image(image2)
st.write('''
There are not many people who have ___lived until the age of 144 or have been employed for 123 years___. These are 
likely outliers we are looking for, because they could negatively affect our model and thus should be removed.
''')

st.code('''
# Scatterplot matrix
sub_data = data[['person_age',"person_income","person_emp_length","loan_amnt","loan_int_rate", "loan_percent_income", 'loan_status']]
sns.pairplot(data=sub_data, hue='loan_status', )
''')
img3 = Image.open('images/img3.png')
st.image(img3)
st.write('It\'s clear that ___Income also has an outlier___.')

st.code('''
# Let's remove outliers from the dataset
# Age > 100, Employment lenght > 100, Income > 4,000,000 
drop_id = data[(data['person_age'] >= 100) | (data['person_emp_length'] >= 100) |
              (data['person_income'] >= 4000000)].index
              
data.drop(index=drop_id, axis=0, inplace=True)
''')
drop_id = data[(data['person_age'] >= 100) | (data['person_emp_length'] >= 100) |
              (data['person_income'] >= 4000000)].index

data.drop(index=drop_id, axis=0, inplace=True)

st.write('''
Looking at the scatter plot, We also find that borrowers, whose loan has ___high interest rate, tend to default___. 
Moreover, person who has ___high loan to income ratio has high chance of default___.
''')

st.code('''
# Check the correlation between numerical variables
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
''')
img4 = Image.open('images/img4.png')
st.image(img4)

st.write('''
We also find that `person_age` and `cb_person_cred_hist_length` ___are highly correlated___. It seems like there is 
multicolinearity problem. Addionally, both variables don't have strong linear relationships with the target variable 
`loan_status`. However, we are not going to drop any column, just yet. \n
Let's check the target variable distribution:
''')

st.code('''
# Check target variable distribution
data.loan_status.value_counts()
''')
st.write(data.loan_status.value_counts())
st.write('''
In this case, we are dealing with an ___imbalanced dataset___, meaning that we have considerably more non-default cases 
than default cases.

### Further explore how loan status is related to other variables
''')

import plotly.express as px
st.code('''
px.box(data, x='loan_grade', y='loan_percent_income', color='loan_status',
       color_discrete_sequence=px.colors.qualitative.Dark24)
''')
fig = px.box(data, x='loan_grade', y='loan_percent_income', color='loan_status',
       color_discrete_sequence=px.colors.qualitative.Dark24)
st.plotly_chart(fig)

st.write('''
When we look at this box plot, there are two things that quickly stand out. We can clearly see that those who don't 
default have a lower loan to income ratio __mean__ value across all `loan grades`. We can also see that no borrowers with 
___loan grade G___ were able to repay their loan.
''')

st.code('''
px.parallel_categories(data_frame=data, dimensions=['person_home_ownership', 'loan_intent', 'loan_grade', 
                                                    'cb_person_default_on_file'], color='loan_status')
''')
fig2 = px.parallel_categories(data_frame=data, dimensions=['person_home_ownership', 'loan_intent', 'loan_grade',
                                                    'cb_person_default_on_file'], color='loan_status')
st.plotly_chart(fig2)
st.write('''
Using a ___parallel category diagram___, we can understand how different categorical variables in our dataset are related to each other and we can map out these relationships on the basis of loan status.

##### Main takeaways from the diagram:
- Our dataset is primarily composed of borrowers who have not defaulted on a loan before.
- Loan grades “A” and “B” are the most common grades while “F” and “G” are the least common;
- Home renters defaulted more often on their loans than those with a mortgage, whereas homeowners defaulted the least;
- Borrowers took out a loan for home improvement the least and for education the most. Also, defaults were more common 
for loans that were taken up for covering medical expenses and debt consolidation.
''')

st.write('''
### See step-by-step model training process
''')
with st.expander('Click to see the code:'):
    st.code('''
# Create a custom feature engineering class
from sklearn.base import BaseEstimator, TransformerMixin

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
        
attrib_adder = CombinedAttributesAdder()
attrib_adder.fit(data)
data = attrib_adder.transform(data)

# Create pipeline to simplify the code, ensure data transformation steps are applied in the correct
# order, and avoid data leakage.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', MinMaxScaler())
])

data['loan_status'] = pd.Categorical(data['loan_status'])
X = data.drop('loan_status', axis=1)
y = data['loan_status']

num_attribs = list(X.select_dtypes(exclude=['object', 'category']).columns)

nominal_cat = list(['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'int_rate_range', 'loan_to_income_range'])

ordinal_cat = ['loan_grade']

trans_pipeline = ColumnTransformer(
    [
        ('ordinal', OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']]), ordinal_cat),
        ('nominal', OneHotEncoder(), nominal_cat),
        ('numeric', num_pipeline, num_attribs)
    ], verbose_feature_names_out=False
)

# Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=.2)

# Create a function to assess the model's ability to predict class labels
from sklearn.metrics import classification_report
def model_assess(model, name):
    print('         ', name, '\n', classification_report(y_test, model.predict(X_test)))

# Import models    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

knn = KNeighborsClassifier(n_neighbors=20)

logreg = LogisticRegression(max_iter=500)

svc = SVC(probability=True)

tree = DecisionTreeClassifier(max_depth=10) # To avoid overfitting

xgb = XGBClassifier()

list_model = [knn, logreg, svc, tree, xgb]
model_names = ['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'XGBClassifier']
diff_model = {}

# Write a function to train and assess the models
def model_training():
    for i, j in enumerate(list_model):
        model_pipeline = Pipeline([
            ('preprocessing', trans_pipeline),
            ('model', j)
        ])
        model_pipeline.fit(X_train, y_train)
        model_assess(model_pipeline, model_names[i])
        diff_model[model_names[i]] = model_pipeline
    return diff_model

# Function to plot ROC_CURVE and AUC scores of each model    
from sklearn.metrics import auc, roc_curve
def plot_roc_curve(y_true, y_probs_dict):
    fig, ax = plt.subplots()
    for name, y_probs in y_probs_dict.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Plot the ROC curve and add labels, title, and legend
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    plt.show()

y_probs_dict = {}
for i in model_names:
    y_probs_dict[i] = trained_models[i].predict_proba(X_test)[:, 1]

plot_roc_curve(y_test, y_probs_dict)

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, random_state=0, stratify=y_train, 
                                                                test_size=.2)

# Perform parameter tuning
from xgboost import XGBClassifier
from sklearn.model_selection importGridSearchCV
from sklearn.model_selection import StratifiedKFold

kfolds = StratifiedKFold(5)

xgb = XGBClassifier(scale_pos_weight=3.5)
param_gird = {
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 1.]
}

model_pipeline = Pipeline([
    ('preprocessing', trans_pipeline),
    ('model', xgb)
])

grid = GridSearchCV(model_pipeline, param_grid=param_gird, scoring='f1', verbose=1, cv=kfolds)
grid.fit(X_train_split, y_train_split)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Validation set score: {:.2f}".format(grid.score(X_val, y_val)))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))

# It seems like the XGBoost model overfit the training data.
# Let's regularize it a bit!

from sklearn.metrics import f1_score
xgb2 = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=150, scale_pos_weight=3.5, reg_alpha=10)

xgb_pipeline = Pipeline([
    ('preprocessing', trans_pipeline),
    ('model', xgb2)
])
xgb_pipeline.fit(X_train, y_train)
y_train_pred = xgb_pipeline.predict(X_train)
print(f1_score(y_train, y_train_pred))

y_test_pred = xgb_pipeline.predict(X_test)
print(f1_score(y_test, y_test_pred))

model_assess(xgb_pipeline, 'XGB')

# Write a function to plot a beautiful confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
plot_confusion_matrix(xgb_pipeline)

# Calculate precision and recall for different probability thresholds
from sklearn.metrics import precision_recall_curve
y_pred_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
precision, recall, threshold = precision_recall_curve(y_test, y_pred_probs)

# Plot precision and recall versus threshold
plt.plot(threshold, precision[:-1], 'b--', label='Precision')
plt.plot(threshold, recall[:-1], 'g-', label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.title(f'Precision and Recall versus Threshold')

# Plot the precision-recall curve
plt.plot(recall, precision, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (XGB)')

# Feature importance plot
from xgboost import plot_importance
xgb_pipeline['model'].get_booster().feature_names = list(xgb_pipeline['preprocessing'].get_feature_names_out())

plot_importance(xgb_pipeline['model'], importance_type='gain', show_values=False, height=0.4)
plt.title('Feature Importance by Information Gain (XGBClassifier)', fontsize = 14)
plt.xlabel('Gain')
    ''')

st.write('#### Summary of models\' performance')
st.markdown('''
This is a case of imbalanced binary classification, because this data contain a small proportion of borrowers who
default on loans (minority class). When dealing with this problem, accuracy is not a reliable performance metric. 
Instead, we will use metrics that are specifically designed to handle imbalanced data such as :green[___precision, recall, F1 
score, and AUC-ROC___].
''')

with st.expander('What is precision and recall?'):
    st.write('''
    Precision measures the proportion of true positives among all predicted positives, and is relevant when the cost of 
    false positives is high (i.e., when you want to minimize the number of false positives). Recall, on the other hand, 
    measures the proportion of true positives among all actual positives, and is relevant when the cost of false n
    egatives is high (i.e., when you want to minimize the number of false negatives).
    ''')

st.markdown('''
In this case, we want to avoid incorrectly classifying an individual who
will default, whereas incorrectly classifying an individual who will not default, though still to be avoided, is less 
problematic.\n
__Recall__ is the most relevant metric because it mesures the proportion of __true positives__ (borrowers who default) 
among __all actual positives__ (all borrowers who should have been classified as high risk).\n

However, a high recall score may also result in a higher number of __false positives__ (i.e., cases where the model 
predicted that a borrower would default, but they did not) due to ___Precision-Recall trade-off___. Therefore, it's 
important to strike a balance between recall and precision, depending on the specific goals of the project and the cost 
associated with different types of errors
''')

st.code('''
          KNeighborsClassifier 
               precision    recall  f1-score   support

           0       0.89      0.97      0.93      4486
           1       0.84      0.58      0.69      1241

    accuracy                           0.89      5727
   macro avg       0.87      0.77      0.81      5727
weighted avg       0.88      0.89      0.88      5727

          LogisticRegression 
               precision    recall  f1-score   support

           0       0.88      0.96      0.91      4486
           1       0.76      0.52      0.62      1241

    accuracy                           0.86      5727
   macro avg       0.82      0.74      0.77      5727
weighted avg       0.85      0.86      0.85      5727

          SVC 
               precision    recall  f1-score   support

           0       0.90      0.98      0.94      4486
           1       0.91      0.61      0.73      1241

    accuracy                           0.90      5727
   macro avg       0.90      0.80      0.83      5727
weighted avg       0.90      0.90      0.89      5727

          DecisionTreeClassifier 
               precision    recall  f1-score   support

           0       0.93      0.99      0.96      4486
           1       0.95      0.72      0.82      1241

    accuracy                           0.93      5727
   macro avg       0.94      0.86      0.89      5727
weighted avg       0.93      0.93      0.93      5727

          XGBClassifier 
               precision    recall  f1-score   support

           0       0.93      0.99      0.96      4486
           1       0.97      0.74      0.84      1241

    accuracy                           0.94      5727
   macro avg       0.95      0.87      0.90      5727
weighted avg       0.94      0.94      0.94      5727

''')

st.markdown('''
We can see that `XGBClassifier` performs better than any other models. Interestingly, a `DecisionTree` classify also does 
fairly good, its __F1__ score is slightly low, compared to `XGBClassifier`'s. Therefore, we can select either one of these to
to arrive at an optimal solution that aligns with our needs and objectives.  
''')


img5 = Image.open('images/img5.png')
st.image(img5)

st.markdown('''
We select `XGBClassifier` for now. The data we are dealing with is imbalanced, so we will remedy this by adding more weight to
the minority class (class 1). After parameter tuning, we get a set of parameters that achieve the highest F1 score:
''')
st.code('''
Fitting 5 folds for each of 27 candidates, totalling 135 fits
Best cross-validation score: 0.82
Validation set score: 0.82
Test set score: 0.83
Best parameters: {'model__learning_rate': 0.1, 'model__max_depth': 7, 'model__n_estimators': 200}
''')

st.write('''
However, the __AUC__ and __F1__ score of the training data are too high, as opposed to those of validation and test set's, 
indicating that the model is overfitting the training data. 
''')
st.code('''
AUC score of training data:  0.9881882967125748
AUC score of validation set:  0.9462456408020923
AUC score of test data 0.9482871413364814
F1 score of training data:  0.9281272300948108
F1 score of validation set:  0.822233875196644
F1 score of test data:  0.825734980826587
''')
st.markdown('''
Let\'s investigate if the model is overfitting. We can plot a `learning curve` which shows the model\'s 'performance
on the training and validation sets as a function of the training set size. The idea behind this is that: if the 
training score is consistently high while the validation score is lower and not improving with more data, it could 
indicate overfitting
''')
img7 = Image.open('images/xgb_learningcurve.png')
st.image(img7)

st.write('''
The fact that the training score is consistently decreasing while the cross-validation score is slowly increasing 
suggests that the model is learning from the training data and improving its performance on the validation set. However,
 the gap between the training and cross-validation scores suggests that there might be some overfitting.
''')

st.write('''
This ___learning curve___ plot suggests that the model is improving its performance on the validation set, but further
investigation is needed to determine if it is overfitting the training data.\n
Let's just __regularize__ the model for now in order to decrease the model complexity and increase generalization power 
when used with new unseen data. 
''')

st.write('###### Final model\'s performance scores:')
st.code('''
          Regularized XGBClassifier 
               precision    recall  f1-score   support

           0       0.95      0.96      0.95      4486
           1       0.84      0.80      0.82      1241

    accuracy                           0.92      5727
   macro avg       0.89      0.88      0.88      5727
weighted avg       0.92      0.92      0.92      5727
''')
st.write('##### Confusion matrix produced by regularized XGBClassifier')
img6 = Image.open('images/img6.png')
st.image(img6)
st.markdown('''
###### This plot shows how many borrowers were correctly or incorrectly classified as high or low risk.

- The "__True Negative__" value of 74.91% means that the model correctly identified a large proportion of low-risk borrowers who did not default on their loans.

- The "__False Negative__" value of 4.28% means that a small proportion of high-risk borrowers who did default on their loans were incorrectly classified as low risk by the model.

- The "__True Positive__" value of 17.39% means that the model correctly identified a small proportion of high-risk borrowers who defaulted on their loans.

- The "__False Positive__" value of 3.42% means that a small proportion of low-risk borrowers who did not default on their loans were incorrectly classified as high risk by the model.

#### In conclusion:
Overall, the model seems to have performed reasonably well in identifying low-risk borrowers who do not default on 
their loans, but has a lower performance in identifying high-risk borrowers who default. This means that the model 
may be more conservative in its risk assessment, and may miss some high-risk borrowers who are likely to default. 
Therefore, further optimization or adjustments may be necessary to improve the model's performance on high-risk 
borrowers. 
''')
