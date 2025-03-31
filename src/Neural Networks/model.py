import time
start_time = time.time()

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data.data_preprossing import data_preprocessing


def NeuralNetwork(NodeCountPerHiddenLayer, ActivationFunction, Optimizer,
                  LearningRateType, MaxIterations, Seed, PrintProgress,
                  X_train, X_test, Y_train, Y_test, MeasureMetrics):

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
        train_accuracy = []
        for i in range(1, MaxIterations + 1):
            Y_train_pred = model.predict(X_train)
            train_accuracy.append(accuracy_score(Y_train, Y_train_pred))

        plt.plot(train_accuracy)
        plt.title("Training Accuracy Progression")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()

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

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print()



# hard to converge with more hidden layers and nodes, and accuracy is the same
if __name__ == "__main__":
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