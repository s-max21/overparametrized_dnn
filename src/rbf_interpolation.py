import numpy as np
from scipy.interpolate import Rbf
from scipy.metrics import mean_squared_error, mean_absolute_error


def train_and_evaluate_rbf(train_data, test_data, function="multiquadric"):
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Create RBF model
    rbf = Rbf(x_train, y_train, function=function)

    # Predict on test data
    y_pred = rbf(x_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae