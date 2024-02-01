import numpy as np
from scipy.interpolate import Rbf
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_and_evaluate_rbf(train_data, test_data, function="multiquadric"):
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Create RBF model
    rbf = Rbf(*x_train, y_train, function=function)

    # Predict on test data
    y_pred = rbf(*x_test)

    # Calculate MSE and MAE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae


def median_and_iqr_rbf(train_data, test_data, samples=50):
    mses = []  # Initialize empty list to store MSEs
    for _ in range(samples):
        mse = train_and_evaluate_rbf(train_data, test_data)[0]
        mses.append(mse)

    return np.median(mses), np.percentile(mses, 75) - np.percentile(mses, 25)