from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_and_evaluate_forest(model, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    model.fit(x_train, y_train)  # Fit the model to training data
    y_pred = model.predict(x_test)  # Predict on test data
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
    return mse, mae

def parameter_tuning_forest(estimators, train_data, test_data):
    best_mse = np.inf  # Set best_mse to infinity
    best_config = None  # Initialize best_config to None
    best_model = None  # Initialize best_model to None

    # Define the parameter grid
    for n_estimators in estimators:
        model = RandomForestRegressor(n_estimators=n_estimators)
        mse, mae = train_and_evaluate_forest(model, train_data, test_data)
        
        if mse < best_mse:  # Note: For MSE, lower is better.
            best_mse = mse
            best_config = n_estimators
            best_model = model
            
        print(f'n_estimators: {n_estimators}, MSE: {mse}')
    
    return best_model, best_config