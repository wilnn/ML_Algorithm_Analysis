from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

def data_loader_and_processor():

    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    x = adult.data.features 
    y = adult.data.targets

    # metadata. Things like dataset name, dataset description, date created, link to the repo of the dataset, etc.,
    #print(adult.metadata) 
    
    # Information for each of the variable (feature) and labels
    # print(adult.variables)

    # replace the NaN (missing value) in each column with the string "Missing". have to do this because
    # pd.factorize will assign the numerical -1 to the NaN missing value which can cause problem, and I want to treat
    # the missing value as one of the categories in those column. 
    x.loc[:, "workclass"] = x["workclass"].fillna("Missing")
    x.loc[:, "occupation"] = x["occupation"].fillna("Missing")
    x.loc[:, "native-country"] = x["native-country"].fillna("Missing")

    # the labels instead of just having 2 category (<=50k, >50k), it also have another  2 categories: '<=50k.' and '>50k.'
    # can be because of typo so need to replace those 2 to make it only have 2 categories
    y.loc[:, "income"] = y["income"].replace({"<=50K.": "<=50K", ">50K.": ">50K"})

    # convert each category in each column to a numerical value and assign it back to that column the dataframe.
    #  print out unique_values1, unique_values2, etc., to see the different category name in those column. 
    x.loc[:, 'workclass'], unique_values1 = pd.factorize(x["workclass"])
    x.loc[:, 'occupation'], unique_values2 = pd.factorize(x["occupation"])
    x.loc[:, 'native-country'], unique_values3 = pd.factorize(x["native-country"])
    x.loc[:, 'education'], unique_values4 = pd.factorize(x["education"])
    x.loc[:, 'marital-status'], unique_values5 = pd.factorize(x["marital-status"])
    x.loc[:, 'relationship'], unique_values6 = pd.factorize(x["relationship"])
    x.loc[:, 'race'], unique_values7 = pd.factorize(x["race"])
    x.loc[:, 'sex'], unique_values8 = pd.factorize(x["sex"])
    y.loc[:, "income"], unique_values9 = pd.factorize(y["income"])


    x = x.iloc[:, :].values
    x = x.astype(np.float32)
    y = y.iloc[:, :].values
    y = y.flatten() # flatten the labels to 1D array since it is 2D after the line above
    y = y.astype(np.int8)

    x_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42) # stratify=y ensure that the class distribution will follow the split ratio. 
    x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    load_dotenv()
    path = os.getenv("DATA_PATH")
    np.save(path + "/processed/not_scaled/train_data.npy", x_train)
    np.save(path + "/processed/not_scaled/train_labels.npy", y_train)
    np.save(path + "/processed/not_scaled/val_data.npy", x_val)
    np.save(path + "/processed/not_scaled/val_labels.npy", y_val)
    np.save(path + "/processed/not_scaled/test_data.npy", x_test)
    np.save(path + "/processed/not_scaled/test_labels.npy", y_test)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x) # scale the data using min max scaling
    
    x_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42) # stratify=y ensure that the class distribution will follow the split ratio. 
    x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    np.save(path + "/processed/min-max_scaled/train_data.npy", x_train)
    np.save(path + "/processed/min-max_scaled/train_labels.npy", y_train)
    np.save(path + "/processed/min-max_scaled/val_data.npy", x_val)
    np.save(path + "/processed/min-max_scaled/val_labels.npy", y_val)
    np.save(path + "/processed/min-max_scaled/test_data.npy", x_test)
    np.save(path + "/processed/min-max_scaled/test_labels.npy", y_test)
    
if __name__ == "__main__":
    data_loader_and_processor()