from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_and_evaluate_tree(tree, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    tree.fit(x_train, y_train)  # Fit the model to training data
    y_pred = tree.predict(x_test)  # Predict on test data
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    return mse, mae


def tune_tree_parameters(train_data):
    x_train, y_train = train_data

    # Define the parameter grid
    param_grid = {
        "max_depth": [None, 5, 10, 15, 20],
        "max_leaf_nodes": [None, 5, 10, 20, 30, 50],
    }

    # Create a DecisionTreeRegressor
    tree = DecisionTreeRegressor()

    # Create a GridSearchCV
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring="neg_mean_squared_error")

    # Fit the GridSearchCV to the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    return best_params


def median_and_iqr_tree(max_depth, max_leaf_nodes, train_data, test_data, samples=50):
    mses = []  # Initialize empty list to store MSEs
    for _ in range(samples):
        model = DecisionTreeRegressor(
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes
        )
        mse = train_and_evaluate_tree(model, train_data, test_data)[0]
        mses.append(mse)

    return np.median(mses), np.percentile(mses, 75) - np.percentile(mses, 25)
