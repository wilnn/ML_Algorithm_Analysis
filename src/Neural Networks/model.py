import time
import pandas as pd
import shap
import numpy as np
from ucimlrepo import fetch_ucirepo

start_time = time.time()

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data.data_preprossing import data_preprocessing


def NeuralNetwork(NodeCountPerHiddenLayer, ActivationFunction, Optimizer,
                  LearningRateType, MaxIterations, Seed, PrintProgress,
                  X_train, X_test, Y_train, Y_test, Feature_names, MeasureMetrics):

    model = MLPClassifier(
        hidden_layer_sizes=NodeCountPerHiddenLayer,
        activation=ActivationFunction,
        solver=Optimizer,
        learning_rate=LearningRateType,
        max_iter=MaxIterations,
        random_state=Seed,
        verbose=PrintProgress
    )
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print(f"Test accuracy: {score * 100:.2f}%")

    if MeasureMetrics:
        plt.plot(model.loss_curve_)
        plt.title("Training Loss Progression")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

        y_pred = model.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()
        print("Classification Report:\n", classification_report(Y_test, y_pred))

        nn_disp = RocCurveDisplay.from_estimator(model, X_test, Y_test)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.show()

        X_background = shap.utils.sample(X_train, 50, random_state=Seed)
        explainer = shap.KernelExplainer(model.predict_proba, X_background)

        X_eval = X_test[:1000]
        shap_values = explainer.shap_values(X_eval, nsamples=100)
        X_eval_df = pd.DataFrame(X_eval, columns=Feature_names)
        shap.summary_plot(shap_values[:, :, 1], X_eval_df, plot_type="bar")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print()



# hard to converge with more hidden layers and nodes, and accuracy is the same
if __name__ == "__main__":
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.drop('education', axis=1)
    feature_names = X.columns.tolist()

    X_train, X_test, Y_train, Y_test = data_preprocessing("StandardScaler")

    NeuralNetwork(
        NodeCountPerHiddenLayer=(15, 15, 15),
        ActivationFunction="relu",
        Optimizer="sgd",
        LearningRateType="constant",
        MaxIterations=500,
        Seed=42,
        PrintProgress=True,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        Feature_names=feature_names,
        MeasureMetrics=True
    )
    NeuralNetwork(
        NodeCountPerHiddenLayer=(50, 50, 50, 50, 50),
        ActivationFunction="logistic",
        Optimizer="adam",
        LearningRateType="adaptive",
        MaxIterations=500,
        Seed=42,
        PrintProgress=True,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        Feature_names=feature_names,
        MeasureMetrics=True
    )
    NeuralNetwork(
        NodeCountPerHiddenLayer=(1000),
        ActivationFunction="tanh",
        Optimizer="sgd",
        LearningRateType="adaptive",
        MaxIterations=500,
        Seed=42,
        PrintProgress=True,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        Feature_names=feature_names,
        MeasureMetrics=True
    )


# MLPClassifier_params = {
#     'hidden_layer_sizes': Tuple of integers indicating the number of neurons in each hidden layer (e.g., (100,), (100, 50))
#     'activation': Activation function for hidden layers ('identity', 'logistic', 'tanh', 'relu')
#     'solver': Solver for weight optimization ('lbfgs', 'sgd', 'adam')
#     'alpha': L2 regularization term (float, default 0.0001)
#     'batch_size': Size of mini-batches for stochastic optimizers ('auto' or an integer)
#     'learning_rate': Learning rate schedule ('constant', 'invscaling', 'adaptive')
#     'learning_rate_init': Initial learning rate for 'sgd' and 'adam' solvers (float)
#     'power_t': Exponent for inverse scaling of learning rate (float)
#     'max_iter': Maximum number of iterations (integer, default 200)
#     'shuffle': Whether to shuffle samples in each iteration (True, False)
#     'random_state': Random seed for reproducibility (integer or None)
#     'tol': Tolerance for optimization stopping criterion (float)
#     'verbose': Whether to display progress messages during training (True, False)
#     'warm_start': Whether to reuse previous solution for initialization (True, False)
#     'momentum': Momentum parameter for 'sgd' solver (float, default 0.9)
#     'nesterovs_momentum': Whether to use Nesterov's momentum (True, False)
#     'early_stopping': Whether to use early stopping based on validation data (True, False)
#     'validation_fraction': Proportion of training data for validation in early stopping (float)
#     'n_iter_no_change': Number of iterations with no improvement before stopping early (integer)
#     'max_fun': Maximum number of function evaluations for 'lbfgs' solver (integer)
# }