from data_preprossing import data_preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

X_train, X_test, y_train, y_test  = data_preprocessing()

svm_model = svm.SVC()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['rbf', 'linear']  # Kernel types
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Train optimized SVM model
best_svm = grid_search.best_estimator_

#save model
dump(best_svm, 'svm_model.joblib')
