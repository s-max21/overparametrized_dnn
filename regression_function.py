import numpy as np
import math


# One dimensional test functions
def m1(x):
    return 2 * (x / 2) ** 3 - 2 * x


def m2(x):
    return 2 * np.abs(x) * np.sin(x * np.pi / 4)


def m3(x):
    if x < 0:
        return -1
    else:
        return 5


def m4(x):
    if x.size[1] != 1:
        raise ValueError("Input must be one dimensional")
    return np.abs((4 * x - 2 * x**2) / (0.5 + x**2))


# Two dimensional test functions
def m5(x):
    if x.size[1] != 2:
        raise ValueError("Input must be two dimensional")
    return x[0] * np.sin(x[0] ** 2) - x[1] * np.sin(x[1] ** 2)


def m6(x):
    if x.size[1] != 2:
        raise ValueError("Input must be two dimensional")
    return 2 / (1 + x[0] ** +x[1] ** 2)


def m7(x):
    if x.size[1] != 2:
        raise ValueError("Input must be two dimensional")
    return 3 - 2 * np.min(3, x[0] ** 2 + np.abs(x[1] ** 2))


def m8(x):
    if x.size[1] != 2:
        raise ValueError("Input must be two dimensional")
    return 1 / (1 + np.exp(-x[0] - x[1]))


# Three dimensional test functions
def m9(x):
    if x.size[1] != 3:
        raise ValueError("Input must be three dimensional")
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2
