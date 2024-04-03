from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_evaluate_knn(model, train_data, test_data):
    """
    Trains the K-Nearest Neighbors model on the training data and evaluates its performance on the test data.

    Args:
        model: K-Nearest Neighbors model
            The model to train and evaluate.
        train_data: tuple
            Tuple of training input data and target values.
        test_data: tuple
            Tuple of test input data and target values.

    Returns:
        float
            Mean squared error between the predicted and actual target values.
    """

    x_train, y_train = train_data
    x_test, y_test = test_data

    model.fit(x_train, y_train)  # Fit the model to training data
    y_pred = model.predict(x_test)  # Predict on test data

    return mean_squared_error(y_test, y_pred)


def parameter_tuning_knn(units, train_data, test_data):
    """
    Tunes the hyperparameters of a K-Nearest Neighbors model to find the best configuration based on the lowest mean squared error (MSE).

    Args:
        units: list
            List of units to tune for the K-Nearest Neighbors model.
        train_data: tuple
            Tuple of training input data and target values.
        test_data: tuple
            Tuple of test input data and target values.

    Returns:
        Tuple of the best K-Nearest Neighbors model, the best number of units, and the corresponding MSE.
    """

    best_mse = np.inf  # Set best_mse to infinity
    best_config = None  # Initialize best_config to None
    best_model = None  # Initialize best_model to None

    for unit in units:
        model = KNeighborsRegressor(n_neighbors=unit)
        mse = train_and_evaluate_knn(model, train_data, test_data)
        print(f"Unit: {unit}, MSE: {mse}, MAE: {mae}")

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_config = unit
            best_model = model

    return best_model, best_config


def generate_neighbors(n):
    """
    Generates a list of neighbors starting from [1, 2, 3] and incrementing by 4 up to 'n'.

    Args:
        n: int
            The upper limit for generating neighbors.

    Returns:
        list
            A list of neighbors starting from [1, 2, 3] and incrementing by 4 up to 'n'.
    """

    u = [1, 2, 3]
    v = list(range(4, n + 1, 4))
    return u + v
