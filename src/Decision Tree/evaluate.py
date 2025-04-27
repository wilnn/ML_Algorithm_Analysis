import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
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

     # ROC Curve
    if len(np.unique(y_test)) == 2:  # binary classification check
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig("results/results/decision_tree_roc_curve.png")
        plt.show()
    else:
        print("ROC Curve skipped (not a binary classification).")

    # SHAP Values (only if input size is reasonable)
    if X_test.shape[0] > 500:
        # To make it fast, sample 500 test examples
        X_sample = shap.sample(X_test, 500, random_state=42)
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plot SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("results/results/decision_tree_shap_summary.png")
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
