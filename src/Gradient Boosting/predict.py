from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from model import gradient_boost_tree
import xgboost as xgb
import sys

model = gradient_boost_tree(n_estimators=127, max_depth=2, learning_rate=1)
s = train_data[:, 2]
s1 = val_data[:, 2]
train_data = np.delete(train_data, 2, axis=1)
val_data = np.delete(val_data, 2, axis=1)
print(train_data.shape)

model.fit(train_data, train_labels, sample_weight=s)
#model.save_model("./Results/models/xgboost_model.json")
# load the best model
#model = xgb.XGBClassifier()
# note that the .load_model will load the data from json file to the model object created earlier.
# doing model = model.load_model("./Results/models/xgboost_model.json") will make the model variable become a nontype object
#model.load_model("./Results/models/xgboost_model.json") 



y_pred = model.predict(val_data)
y_pred = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(val_labels, y_pred)
print(f"accuracy: {accuracy}")

precision = precision_score(val_labels, y_pred)
recall = recall_score(val_labels, y_pred)
f1 = f1_score(val_labels, y_pred)
print(f"F1: {f1:.2f}") # average f1 of all classes
print(f"Precision: {precision:.2f}") # average precision of all classes
print(f"Recall: {recall:.2f}") # average recall of all classes

print("Result on the train set (to check bias and overfitting):")
# check result for train set
y_pred = model.predict(train_data)
y_pred = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(train_labels, y_pred)
print(f"accuracy: {accuracy}")

precision = precision_score(train_labels, y_pred)
recall = recall_score(train_labels, y_pred)
f1 = f1_score(train_labels, y_pred)
print(f"F1: {f1:.2f}") # average f1 of all classes
print(f"Precision: {precision:.2f}") # average precision of all classes
print(f"Recall: {recall:.2f}") # average recall of all classes
