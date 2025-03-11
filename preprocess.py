import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    columns = [
        'Gender_Marital', 'Age', 'Debt', 'Purpose', 'Employment_Status', 'Credit_History',
        'Housing', 'Monthly_Income', 'Checking_Account', 'Savings_Account', 'Credit_Lines',
        'Guarantor', 'Property_Ownership', 'Years_At_Job', 'Existing_Loan_Amount', 'TARGET'
    ]
    data = pd.read_csv(data_url, header=None, names=columns, na_values='?')
    data['TARGET'] = data['TARGET'].map({'+': 1, '-': 0})  # Convert TARGET to binary

    # Identify categorical and numerical columns
    cat_cols = data.select_dtypes(include=['object']).columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop('TARGET')

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
    scaler = StandardScaler()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), num_cols),
        ('cat', Pipeline([('imputer', cat_imputer), ('encoder', encoder)]), cat_cols)
    ])

    X = data.drop(columns=['TARGET'])
    y = data['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    preprocessor.fit(X_train)

    # Return cat_cols along with other variables
    return X_train, X_test, y_train, y_test, preprocessor, cat_cols
