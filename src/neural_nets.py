import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from data.data_generator import get_data, preprocess


def create_neural_1(input_dim, units=64, activation="relu"):
    """
    Creates a neural network with one hidden layers and one output layer.
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(units, activation=activation),
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
            Input(shape=(input_dim,)),
            Dense(units, activation=activation),
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
            Input(shape=(input_dim,)),
            Dense(units, activation=activation),
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


def train_and_evaluate_nn(model, train_data, test_data, epochs=75):
    """
    Trains the model on the given data and evaluates its performance.
    """
    model.fit(train_data, epochs=epochs, verbose=0)
    mse, mae = model.evaluate(test_data, verbose=0)
    return mse, mae


def parameter_tuning_nn(create_neural_network, units, train_data, test_data, input_dim, epochs):
    """
    Tunes the model's parameters to find the best hyperparameters.
    """
    best_mse = np.inf  # Set best_mse to infinity
    best_hp = None  # Initialize best_hp to None

    for unit in units:
        model = create_neural_network(input_dim=input_dim, units=unit)
        mse, mae = train_and_evaluate_nn(model, train_data, test_data, epochs=epochs)
        print(f"Unit: {unit}, MSE: {mse}, MAE: {mae}")

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_hp = unit
    return {"best_hp": best_hp, "mse": best_mse}


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
            regression_func, x_dim=input_dim, num_samples=100, sigma=0.05
        )
        x_test, y_test = get_data(
            regression_func, x_dim=input_dim, num_samples=10**5, sigma=0.05
        )

        # Preprocess data
        train_data = preprocess(x_train, y_train, batch_size=batch_size, training=True)
        test_data = preprocess(x_test, y_test, batch_size=batch_size, training=False)

        model = create_neural_network(input_dim=input_dim, units=units)
        mse, mae = train_and_evaluate_nn(
            model, train_data, test_data, epochs=epochs
        )
        mses.append(mse)
        maes.append(mae)

    return mses, maes
