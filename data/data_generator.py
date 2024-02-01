# -*- coding: utf-8 -*-
"""data_generator.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rMBPY4scSpAoKeY6TXfWYbP9XPa7oeZe
"""

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
    # Calculate the median of the IQR values
    median_iqr = np.median(iqr_values)

    return median_iqr


# Generate random data samples
def get_data(my_func, x_dim=1, num_samples=10**2, sigma=0.05):
    omega = iqr_median(my_func, x_dim=x_dim)

    x = np.random.rand(num_samples, x_dim)
    y = my_func(x) + sigma * omega * np.random.normal(0, 1, (num_samples,))

    return (x, y)


# Optimized data loading
def preprocess(x, y, batch_size=64, training=False):
    data = tf.data.Dataset.from_tensor_slices((x, y))

    if training:
        data = data.shuffle(buffer_size=len(x))

    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data
