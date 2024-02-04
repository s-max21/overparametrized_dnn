import numpy as np
from scipy.interpolate import Rbf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.data_generator import get_data


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


def median_and_iqr_rbf(input_dim, regression_func, samples=50):
    mses = []  # Initialize empty list to store MSEs
    for _ in range(samples):
        x_train, y_train = get_data(
            regression_func, x_dim=input_dim, num_samples=1000, sigma=0.05
        )
        x_test, y_test = get_data(
            regression_func, x_dim=input_dim, num_samples=10**5, sigma=0.05
        )

        # Preprocess data
        train_data = (x_train, y_train)
        test_data = (x_test, y_test)

        mse = train_and_evaluate_rbf(train_data, test_data)[0]
        mses.append(mse)

    return np.median(mses), np.percentile(mses, 75) - np.percentile(mses, 25)
