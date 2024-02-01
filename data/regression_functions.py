import numpy as np


# One dimensional test functions
def m1(x):
    return np.abs((4 * x - 2 * x**2) / (0.5 + x**2))


def m2(x):
    return np.sin(2 * np.pi * x) + 0.5 * x**2 + 0.2 * x + 0.1 * np.sin(5 * np.pi * x)


def m3(x):
    if x < 0.5:
        return -1
    else:
        return 5


# Two dimensional test function
def m4(x):
    return x[0] * np.sin(x[0] ** 2) - x[1] * np.sin(x[1] ** 2)


# Three dimensional test function
def m5(x):
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2


# Four dimensional test function
def m6(x):
    return 2 / (x[0] + 0.008) + 3 * np.log(x[1] ** 7 * x[2] + 0.1) * x[3]


# Seven dimensional test function
def m7(x):
    return np.arctan(
        np.pi / (1 + np.exp(x[0] ** 2 + 2 * x[1] + np.sin(6 * x[3] ** 3) - 3))
    ) + np.exp(3 * x[2] + 2 * x[3] - 5 * x[4] + np.sqrt(x[5] + 0.9 * x[6] + 0.1))


# N-dim test function
def m8(x):
    return np.exp(np.linalg.norm(x, axis=1))
