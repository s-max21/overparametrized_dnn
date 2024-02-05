from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.data_generator import get_data


def train_and_evaluate_tree(tree, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    tree.fit(x_train, y_train)  # Fit the model to training data
    y_pred = tree.predict(x_test)  # Predict on test data
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    return mse, mae


def parameter_tuning_tree(train_data):
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


def runs_tree(
    max_depth, max_leaf_nodes, input_dim, regression_func, samples=50
):
    mses = []  # Initialize empty list to store MSEs
    maes = []  # Initialize empty list to store MAEs
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

        model = DecisionTreeRegressor(
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes
        )
        mse, mae = train_and_evaluate_tree(model, train_data, test_data)
        mses.append(mse)
        maes.append(mae)

    return mses, maes
