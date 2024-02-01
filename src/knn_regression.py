from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_and_evaluate_knn(model, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    model.fit(x_train, y_train)  # Fit the model to training data
    y_pred = model.predict(x_test)  # Predict on test data
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    return mse, mae


def parameter_tuning_knn(units, train_data, test_data):
    best_mse = np.inf  # Set best_mse to infinity
    best_config = None  # Initialize best_config to None

    for unit in units:
        model = KNeighborsRegressor(n_neighbors=unit)
        mse, mae = train_and_evaluate_knn(model, train_data, test_data)
        print(f"Unit: {unit}, MSE: {mse}, MAE: {mae}")

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_config = unit
    return {"best_config": best_config, "mse": mse}


def median_and_iqr_knn(train_data, test_data, unit, samples=50):
    mses = []  # Initialize empty list to store MSEs
    for _ in range(samples):
        model = KNeighborsRegressor(n_neighbors=unit)
        mse, mae = train_and_evaluate_knn(model, train_data, test_data)[0]
        mses.append(mse)

    return np.median(mses), np.percentile(mses, 75) - np.percentile(mses, 25)
