import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
import shap

# Load the data
val_data = np.load("./data/processed/not_scaled/val_data.npy")
val_labels = np.load("./data/processed/not_scaled/val_labels.npy")
test_data = np.load("./data/processed/not_scaled/test_data.npy")
test_labels = np.load("./data/processed/not_scaled/test_labels.npy")

# Concatenate validation and test data
testx = np.concatenate((val_data, test_data), axis=0)
testy = np.concatenate((val_labels, test_labels), axis=0)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("./Results/models/xgboost_model.json")

def compute_shap_and_visualize(model, test_data):
    """
    Function to compute SHAP values and visualize them for a given model and test data.
    
    Parameters:
    - model: The trained machine learning model (e.g., XGBClassifier).
    - test_data: The test dataset to explain with SHAP.
    """
    # Create SHAP explainer for the model
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(test_data)
    
    # Display SHAP summary plot
    print("Displaying SHAP Summary Plot...")
    shap.summary_plot(shap_values, test_data, feature_names=['age', 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
    
    # Display SHAP feature importance as bar plot for easier understanding
    print("Displaying SHAP Feature Importance Bar Plot...")
    shap.summary_plot(shap_values, test_data, plot_type="bar", feature_names=['age', 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
    
    # Display SHAP force plot for the first sample in the test set
    # A SHAP force plot is a visualization that helps explain how individual features contribute to a model's prediction for a specific data point. 
    shap.force_plot(explainer.expected_value, shap_values[0], test_data[0,:])

    plt.savefig("./src/Gradient Boosting/shap_force_plot.png")
    plt.close()
    # Display SHAP decision plot for the first few samples
    print("Displaying SHAP Decision Plot...")
    shap.decision_plot(explainer.expected_value, shap_values, test_data, ignore_warnings=True)
    plt.savefig("./src/Gradient Boosting/shap_decision_plot.png")
    plt.close()
    print("SHAP Decision Plot saved as shap_decision_plot.html")

# Function to compute and plot confusion matrix
def plot_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Function to compute and plot precision score
def compute_precision_score(model, X, y):
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred, pos_label=0)
    print(f"Precision Score for predicting <=50k: {precision:.4f}")
    precision = precision_score(y, y_pred, pos_label=1)
    print(f"Precision Score for predicting >50k: {precision:.4f}")


# Function to compute and plot accuracy as a pie chart
def plot_accuracy_pie_chart(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    plt.figure(figsize=(5, 5))
    plt.pie([accuracy, 1 - accuracy], labels=["Correct", "Incorrect"], autopct='%1.1f%%', colors=["green", "red"])
    plt.title(f"Accuracy: {accuracy:.2f}")
    plt.show()


# Function to plot ROC curve and display TPR, FPR
def plot_roc_curve(model, X, y):
    # Binarize the output for ROC
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_bin, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.show()

# Function to compute recall score
def compute_recall(model, data, labels):
    predictions = model.predict(data)
    recall = recall_score(labels, predictions, pos_label=0)
    print(f"Recall Score for predicting <=50k: {recall:.4f}")
    recall = recall_score(labels, predictions, pos_label=1)
    print(f"Recall Score for predicting >50k: {recall:.4f}")
    return recall

# Function to compute F1 score
def compute_f1(model, data, labels):
    predictions = model.predict(data)
    f1 = f1_score(labels, predictions, pos_label=0)
    print(f"F1 Score for predicting <=50k: {f1:.4f}")
    f1 = f1_score(labels, predictions, pos_label=1)
    print(f"F1 Score for predicting >50k: {f1:.4f}")
    return f1

# Call the functions with the model and test data
plot_confusion_matrix(model, testx, testy)
compute_precision_score(model, testx, testy)
plot_accuracy_pie_chart(model, testx, testy)
plot_roc_curve(model, testx, testy)
compute_f1(model, testx, testy)
compute_recall(model, testx, testy)

compute_shap_and_visualize(model, testx)
