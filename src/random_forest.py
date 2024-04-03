from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_and_evaluate_forest(model, train_data, test_data):
    """
    Trains the given Random Forest on the training data and evaluates its performance on the test data.

    Args:
        model: Random Forest model
            The random forest model to train and evaluate.
        train_data: tuple
            Tuple of training input data and target values.
        test_data: tuple
            Tuple of test input data and target values.

    Returns:
        Tuple of mean squared error (MSE) between the predicted and actual target values.
    """

    x_train, y_train = train_data
    x_test, y_test = test_data

    model.fit(x_train, y_train)  # Fit the model to training data
    y_pred = model.predict(x_test)  # Predict on test data

    return mean_squared_error(y_test, y_pred)


def parameter_tuning_forest(estimators, train_data, test_data):
    """
    Performs parameter tuning for a Random Forest model by iterating over different numbers of estimators.
    Selects the best model configuration based on the lowest mean squared error (MSE) on the test data.

    Args:
        estimators: list
            List of the number of estimators to tune.
        train_data: tuple
            Tuple of training input data and target values.
        test_data: tuple
            Tuple of test input data and target values.

    Returns:
        Tuple of the best Random Forest model, the best number of estimators, and the corresponding MSE.
    """

    best_mse = np.inf  # Set best_mse to infinity
    best_config = None  # Initialize best_config to None
    best_model = None  # Initialize best_model to None

    # Define the parameter grid
    for n_estimators in estimators:
        model = RandomForestRegressor(n_estimators=n_estimators)
        mse = train_and_evaluate_forest(model, train_data, test_data)

        # Update best model if current MSE is lower
        if mse < best_mse:
            best_mse = mse
            best_config = n_estimators
            best_model = model

    return best_model, best_config