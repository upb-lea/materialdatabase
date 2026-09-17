"""Temperature-dependent empirical models."""

import numpy as np


def quadratic_temperature(T: float | np.ndarray, c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Quadratic temperature dependence: k(T) = c_0 - c_1*T + c_2*T^2.

    :param T: Temperature in deg C or K
    :param c_0: Constant coefficient
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Temperature-dependent scaling factor
    """
    return c_0 - c_1 * T + c_2 * T ** 2