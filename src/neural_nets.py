import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential


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
        name="neural-1",
    )

    model.compile(optimizer="adam", loss="mse")

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
        name="neural-3",
    )

    model.compile(optimizer="adam", loss="mse")

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
        name="neural-6",
    )

    model.compile(optimizer="adam", loss="mse")

    return model


def train_and_evaluate_nn(model, train_data, test_data, epochs=75):
    """
    Trains the model on the given data and evaluates its performance.
    """
    model.fit(train_data, epochs=epochs, verbose=0)
    return model.evaluate(test_data, verbose=0)


def parameter_tuning_nn(
    create_neural_network, units, train_data, test_data, input_dim, epochs
):
    """
    Tunes the model's hyperparameters to find the best configuration based on the lowest mean squared error (MSE).

    Parameters
    ----------
    create_neural_network: function
        Function to create a neural network model.
    units: list
        List of units to tune for the neural network.
    train_data: tuple
        Tuple of training input data and target values.
    test_data: tuple
        Tuple of test input data and target values.
    input_dim: int
        Dimension of the input data.
    epochs: int
        Number of epochs for training.

    Returns
    -------
    Tuple of the best neural network model, the best number of units, and the corresponding MSE.
    """

    best_mse = np.inf  # Set best_mse to infinity
    best_hp = None  # Initialize best_hp to None
    best_model = None  # Initialize best_model to None

    for unit in units:
        model = create_neural_network(input_dim=input_dim, units=unit)
        mse = train_and_evaluate_nn(model, train_data, test_data, epochs=epochs)

        # Check if current MSE is better than the best MSE so far
        if best_mse > mse:
            best_mse = mse
            best_hp = unit
            best_model = model

    return best_model, best_hp