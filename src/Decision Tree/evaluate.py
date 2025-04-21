import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

def evaluate():
    # Load test data and labels from .npy files
    X_test = np.load("data/processed/not_scaled/test_data.npy")
    y_test = np.load("data/processed/not_scaled/test_labels.npy")

    # Load trained model
    model = joblib.load("results/models/decision_tree.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save evaluation metrics
    with open("results/results/decision_tree_evaluation.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)
    
    print(f"Accuracy: {accuracy}")
    print(report)

    # 🔷 Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/results/confusion_matrix.png")
    plt.show()

    # 🔷 Visualize the Decision Tree
    # If your model was trained on a DataFrame, feature names might be available
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("results/results/decision_tree_structure.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
