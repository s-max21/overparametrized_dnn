import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def create_network_1(input_dim, units=64, activation="relu"):
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(1, activation="linear"),
        ]
    )
    return model


def create_network_2(input_dim, units=64, activation="relu"):
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(1, activation="linear"),
        ]
    )
    return model


def create_network_3(input_dim, units=64, activation="relu"):
    model = Sequential(
        [
            Dense(units, activation=activation, input_shape=(input_dim,)),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(units, activation=activation),
            Dense(1, activation="linear"),
        ]
    )
    return model


def train_and_evaluate_nn(model, train_data, test_data):
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(train_data, epochs=15, batch_size=32, verbose=0)
    mse, mae = model.evaluate(test_data, verbose=0)
    return mse, mae


def parameter_tuning_nn(create_network, units, train_data, test_data, input_dim):
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


def median_and_iqr_nn(
    create_network, train_data, test_data, input_dim, units=64, samples=50
):
    mses = []  # Initialize empty list to store MSEs
    maes = []  # Initialize empty list to store MAEs
    for _ in range(samples):
        model = create_network(input_dim=input_dim, units=units)
        mse, mae = train_and_evaluate_nn(model, train_data, test_data)
        mses.append(mse)
        maes.append(mae)
        
    return {
        "median_mse": np.median(mses),
        "median_mae": np.median(maes),
        "iqr_mse": np.percentile(mses, 75) - np.percentile(mses, 25),
        "iqr_mae": np.percentile(maes, 75) - np.percentile(maes, 25),
    }
