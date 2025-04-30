import joblib

def load_model(model_path="results/models/decision_tree.pkl"):
    """Load the trained model."""
    return joblib.load(model_path)
