import numpy as np
import joblib
from model import get_model

def train():
    # Load processed data from .npy files
    X_train = np.load("data/processed/not_scaled/train_data.npy")
    y_train = np.load("data/processed/not_scaled/train_labels.npy")

    # Get model and train
    model = get_model()
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "results/models/decision_tree.pkl")
    print("Model training completed and saved.")

if __name__ == "__main__":
    train()
