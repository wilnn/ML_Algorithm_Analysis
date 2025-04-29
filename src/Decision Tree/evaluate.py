import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree

# Actual feature names for the Adult dataset (Census Income dataset)
feature_names_adult = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"
]

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
    with open("./src/Decision Tree/decision_tree_evaluation.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)
    
    print(f"Accuracy: {accuracy}")
    print(report)

     # Plot Accuracy Pie Chart
    correct = np.sum(y_pred == y_test)
    incorrect = np.sum(y_pred != y_test)

    plt.figure(figsize=(6, 6))
    plt.pie(
        [correct, incorrect],
        labels=["Correct", "Incorrect"],
        colors=["#4CAF50", "#F44336"],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),
        shadow=True
    )
    plt.title("Prediction Accuracy")
    plt.savefig("./src/Decision Tree/accuracy_pie_chart.png")
    plt.show()
    
    # 🔷 Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("./src/Decision Tree/confusion_matrix.png")
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
        plt.savefig("./src/Decision Tree/decision_tree_roc_curve.png")
        plt.show()
    else:
        print("ROC Curve skipped (not a binary classification).")


    # --- SHAP EVALUATION ---
    print("Running SHAP evaluation...")

    # Check if the model has the 'feature_names_in_' attribute
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        # In case 'feature_names_in_' is not available, generate default names
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
     # Use actual feature names (provided in 'feature_names_adult')
    feature_names = feature_names_adult  # Assign actual feature names for the Adult dataset

    # Convert to DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Use TreeExplainer explicitly
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # Check if SHAP values are a list (for multiclass)
    if isinstance(shap_values, list):
        print("Multiclass classification detected.")
        # Select SHAP values for class 1 (positive class)
        shap_values_to_plot = shap_values[1]  # class 1 SHAP values
    else:
        print("Binary classification detected.")
        shap_values_to_plot = shap_values  # SHAP values for binary classification

    # Ensure SHAP values are 3D (samples x features x classes)
    if shap_values_to_plot.ndim == 3:
        # Take SHAP values for the positive class (usually class 1 in binary classification)
        shap_values_to_plot = shap_values_to_plot[:, :, 1]

    # Ensure shap_values_to_plot is 2D
    if shap_values_to_plot.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, but got {shap_values_to_plot.ndim}D.")

    # SHAP summary plot (Feature Importance)
    shap.summary_plot(shap_values_to_plot, X_test_df, show=True)

        # --- Feature Importance ---
    # Calculate feature importance as the mean absolute SHAP value across all samples
    feature_importance = np.abs(shap_values_to_plot).mean(axis=0)
    feature_importance_df = pd.DataFrame(
        {'Feature': feature_names, 'Importance': feature_importance}
    ).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance (based on SHAP values):")
    print(feature_importance_df)

    # --- Feature Contribution Bar Plot ---
    # Plot the bar plot for feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title("Feature Importance (based on SHAP values)")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("./src/Decision Tree/feature_importance_bar_plot.png")
    plt.show()

    # --- Feature Contribution (SHAP Force Plot) ---
    shap.initjs()  # Initialize JavaScript for visualizations

    sample_idx = 1  # Choose a sample (you can choose other samples as needed)

    # Create force plot for the first sample
    force_plot = shap.force_plot(
        explainer.expected_value[1],  # Expected value for class 1 (positive class)
        shap_values_to_plot[sample_idx],  # SHAP values for the selected sample
        X_test_df.iloc[sample_idx],  # Feature values for the selected sample
    )
    shap.save_html("./src/Decision Tree/shap_force_plot.html", force_plot)
    print("SHAP force plot saved to './src/Decision Tree/shap_force_plot.html'")

    # 🔷 Visualize the Decision Tree
    # If your model was trained on a DataFrame, feature names might be available
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("./src/Decision Tree/decision_tree_structure.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
