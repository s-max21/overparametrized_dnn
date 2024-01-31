import tensorflow as tf
from tf.keras.layers import Dense


def create_network_1():
    model = tf.keras.models.Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dense(1, activation="linear"),
        ]
    )
    return model


def create_network_2():
    model = tf.keras.models.Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    return model


def create_network_3():
    model = tf.keras.models.Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(2, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    return model
