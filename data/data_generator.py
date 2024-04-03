import numpy as np
import tensorflow as tf


# Function to calculate IQR
def calculate_iqr(data):

    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25


def iqr_median(my_func, x_dim=1, num_samples=10**5, num_repetitions=100):
    """Function to calculate the IQR for a given function"""

    iqr_values = []
    # Generate random samples of x and calculate IQR for each set
    for _ in range(num_repetitions):
        x_samples = np.random.rand(num_samples, x_dim)
        my_func_values = my_func(x_samples)
        iqr_values.append(calculate_iqr(my_func_values))

    return np.median(iqr_values)


# Generate random data samples
def get_data(my_func, x_dim=1, num_samples=10**3, sigma=0.05, omega=None, seed=42):
    """Function to generate random data samples"""

    rng = np.random.default_rng(seed)

    if omega is None:
        omega = iqr_median(my_func, x_dim=x_dim)

    x = rng.random(size=(num_samples, x_dim))
    y = my_func(x) + sigma * omega * rng.standard_normal(size=num_samples)

    return (x, y)


# Optimized data loading
def preprocess(x, y, batch_size=64, training=False):
    """Function to preprocess data for training"""

    data = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        data = data.shuffle(buffer_size=len(x))

    data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data
