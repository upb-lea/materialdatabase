"""Mathmatical functions."""
import numpy as np


def mre(x: float | np.ndarray, x_est: float | np.ndarray) -> float:
    """
    Calculate mean relative absolute error.

    :param x: ground truth value
    :param x_est: estimated value
    :return: mean relative absolute error
    """
    return float(np.mean(abs((x - x_est) / x)))
