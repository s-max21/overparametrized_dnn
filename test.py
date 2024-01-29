from data_generator import get_data, preprocess
import numpy as np 

def my_func(x):
    return np.sin(2 * np.pi * x)

x, y = get_data(my_func)
data = preprocess(x, y, batch_size=64, training=True)
print(data)
