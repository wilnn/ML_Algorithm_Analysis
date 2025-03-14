from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

def data_preprocessing(scaler = "StandardScaler"):
    # Fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # Data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets

    # Reaplce '?' to NaN, and clean the data
    combined = pd.concat([X, y], axis=1).replace('?', pd.NA)
    combined = combined.dropna()
    X = combined.drop('income', axis=1)
    y = combined['income']
    y = y.replace({"<=50K.": "<=50K", ">50K.": ">50K"})

    # Since 'education' and ‘education_num' are the same thing, we remove one of them
    X = X.drop('education', axis=1)

    # One-hot encode
    for col in X.select_dtypes('object').columns:
        X[col] = LabelEncoder().fit_transform(y=X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Decision Trees and Gradient Boosting methods are not affected by feature scaling.
    # Neural Networks perform better when inputs are normalized or standardized. Both can be used
    # Support Vector Machines (SVM) use StandardScaler in general, and use RobustScaler if your data contains significant outliers
    # If there are other Scaler you want to use, you can add it on there.
    if scaler == "MinMaxScaler":
        sc = MinMaxScaler()
    elif scaler == "StandardScaler":
        sc = StandardScaler()
    else: 
        return X_train, X_test, y_train, y_test
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test