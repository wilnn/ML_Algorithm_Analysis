import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def predict():
    # Load test data and labels
    X_test = np.load("data/processed/not_scaled/test_data.npy")
    y_test = np.load("data/processed/not_scaled/test_labels.npy")  # Optional, for evaluation

    # Load trained model
    model = joblib.load("results/models/decision_tree.pkl")

    # Make predictions
    predictions = model.predict(X_test)

    # Save predictions
    pd.DataFrame(predictions, columns=["Predicted"]).to_csv("results/results/decision_tree_predictions.csv", index=False)
    print("Predictions saved to 'results/results/decision_tree_predictions.csv'.")

    '''# Optional: Evaluate the model

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))'''

if __name__ == "__main__":
    predict()
