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
    maes = []  # Initialize empty list to store MAEs
    for _ in range(samples):
        mse, mae = train_and_evaluate_rbf(train_data, test_data)
        mses.append(mse)
        maes.append(mae)

    return {
        "median_mse": np.median(mses),
        "median_mae": np.median(maes),
        "iqr_mse": np.percentile(mses, 75) - np.percentile(mses, 25),
        "iqr_mae": np.percentile(maes, 75) - np.percentile(maes, 25),
    }
