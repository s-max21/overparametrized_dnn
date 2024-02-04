import numpy as np


def add_expected_dim(dim):
    def decorator(func):
        func.expected_dim = dim
        return func

    return decorator


# One dimensional test functions
@add_expected_dim(1)
def m1(x):
    return np.squeeze(np.abs((4 * x - 2 * x**2) / (0.5 + x**2)))


@add_expected_dim(1)
def m2(x):
    return np.squeeze(
        np.sin(2 * np.pi * x) + 0.5 * x**2 + 0.2 * x + 0.1 * np.sin(5 * np.pi * x)
    )


@add_expected_dim(1)
def m3(x):
    return np.squeeze(np.where(x > 0.5, 5, -1))


# Two dimensional test function
@add_expected_dim(2)
def m4(x):
    return x[:, 0] * np.sin(x[:, 0] ** 2) - x[:, 1] * np.sin(x[:, 1] ** 2)


# Three dimensional test function
@add_expected_dim(3)
def m5(x):
    return (
        2 * x[:, 0] ** 2
        - 1.05 * x[:, 2] ** 4
        + x[:, 0] ** 6 / 6
        + x[:, 0] * x[:, 1]
        + x[:, 1] ** 2
    )


# Four dimensional test function
@add_expected_dim(4)
def m6(x):
    return 2 / (x[:, 0] + 0.008) + 3 * np.log(x[:, 1] ** 7 * x[:, 2] + 0.1) * x[:, 3]


# Seven dimensional test function
@add_expected_dim(7)
def m7(x):
    return np.arctan(
        np.pi / (1 + np.exp(x[:, 0] ** 2 + 2 * x[:, 1] + np.sin(6 * x[:, 3] ** 3) - 3))
    ) + np.exp(
        3 * x[:, 2] + 2 * x[:, 3] - 5 * x[:, 4] + np.sqrt(x[:, 5] + 0.9 * x[:, 6] + 0.1)
    )


@add_expected_dim(7)
def m8(x):
    return np.exp(np.linalg.norm(x, axis=1))
