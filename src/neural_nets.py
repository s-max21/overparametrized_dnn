import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import keras_tuner as kt
from data.data_generator import get_data, preprocess


def create_neural_1(input_dim, units=64, activation="relu"):
    """
    Creates a neural network with one hidden layers and one output layer.
    """
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(1, activation="linear"),
        ],
        name="one_hidden_layer",
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"])
    
    return model


def create_neural_3(input_dim, units=64, activation="relu"):
    """
    Creates a neural network with three hidden layers and one output layer.
    """
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(1, activation="linear"),
        ],
        name="three_hidden_layers",
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"])
    
    return model


def create_neural_6(input_dim, units=64, activation="relu"):
    """
    Creates a neural network with six hidden layers and one output layer.
    """
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(1, activation="linear"),
        ],
        name="six_hidden_layers",
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"])
    
    return model


def train_and_evaluate_nn(model, train_data, test_data, epochs=15, batch_size=32):
    """
    Trains the model on the given data and evaluates its performance.
    """
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(train_data, epochs=epochs, batch_size=batch_size, verbose=0)
    mse, mae = model.evaluate(test_data, verbose=0)
    return mse, mae


def parameter_tuning_nn(create_network, units, train_data, test_data, input_dim):
    """
    Tunes the model's parameters to find the best configuration.
    """
    best_mse = np.inf  # Set best_mse to infinity
    best_config = None  # Initialize best_config to None

    for unit in units:
        model = create_network(input_dim=input_dim, units=unit)
        mse, mae = train_and_evaluate_nn(model, train_data, test_data)
        print(f"Unit: {unit}, MSE: {mse}, MAE: {mae}")

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_config = unit
    return {"best_config": best_config, "mse": mse}


def parameter_tuning(create_neural_network, train_data, val_data, input_dim, min_value, max_value, step):

    # Define the model-building function with a single hyperparameter for all layers
    def build_model(hp):
        hp_units = hp.Int('units', min_value=min_value, max_value=max_value, step=step)
        model = create_neural_network(units = hp_units, input_dim=input_dim)

        return model

    # Initialize the tuner
    tuner = kt.GridSearch(
        build_model,
        objective='val_mean_squared_error' # Objective is to minimize validation mean squared error
        )  
    

    # Start the hyperparameter search
    tuner.search(train_data, epochs=100, validation_data=val_data)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]

    return best_hps


def runs_nn(
    create_neural_network,
    input_dim,
    regression_func,
    units,
    samples=50,
    epochs=15,
    batch_size=32,
):
    """
    Calculates the median and interquartile range of the model's performance.
    """
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
        train_data = preprocess(x_train, y_train, training=True)
        test_data = preprocess(x_test, y_test, training=False)

        model = create_neural_network(input_dim=input_dim, units=units)
        mse, mae = train_and_evaluate_nn(
            model, train_data, test_data, epochs=epochs, batch_size=batch_size
        )
        mses.append(mse)
        maes.append(mae)

    return mses, maes
