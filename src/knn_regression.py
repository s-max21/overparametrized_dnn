from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.data_generator import get_data
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
    best_model = None  # Initialize best_model to None

    for unit in units:
        model = KNeighborsRegressor(n_neighbors=unit)
        mse, mae = train_and_evaluate_knn(model, train_data, test_data)
        print(f"Unit: {unit}, MSE: {mse}, MAE: {mae}")

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_config = unit
            best_model = model
            
    return best_model, best_config


def generate_neighbors():
    u = [1, 2, 3]
    v = list(range(4, x_train.shape[0]+1, 4))
    return u+v


def runs_knn(unit, input_dim, regression_func, samples=50):
    mses = []  # Initialize empty list to store MSEs
    maes = []  # Initialize empty list to store MAEs
    for _ in range(samples):
        x_train, y_train = get_data(
            regression_func, x_dim=input_dim, num_samples=100, sigma=0.05
        )
        x_test, y_test = get_data(
            regression_func, x_dim=input_dim, num_samples=10**5, sigma=0.05
        )

        # Preprocess data
        train_data = (x_train, y_train)
        test_data = (x_test, y_test)

        model = KNeighborsRegressor(n_neighbors=unit)
        mse, mae = train_and_evaluate_knn(model, train_data, test_data)
        mses.append(mse)
        maes.append(mae)

    return mses, maes
