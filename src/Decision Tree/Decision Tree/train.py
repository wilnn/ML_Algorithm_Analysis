import numpy as np
import joblib
from model import get_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from data_preprossing import data_preprocessing

def train():

    # Get preprocessed data
    X_train, X_test, y_train, y_test = data_preprocessing()



    # Get model and train
    model = get_model()
    model.fit(X_train, y_train)

    '''# Save trained model
    joblib.dump(model, "results/models/decision_tree.pkl")
    print("Model training completed and saved.")'''

    # Define the grid of hyperparameters you want to search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        n_jobs=-1,           # Use all CPUs
        verbose=2,
        scoring='accuracy'   # Optimize for accuracy
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best model from the search
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, "results/models/decision_tree.pkl")
    print("Best Model saved to 'results/models/decision_tree.pkl'.")

    # Also print best parameters
    print("\nBest Parameters Found:")
    print(grid_search.best_params_)

if __name__ == "__main__":
    train()
