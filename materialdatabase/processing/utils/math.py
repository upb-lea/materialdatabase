"""Mathmatical functions."""
import numpy as np
from typing import Any


def mre(x: float | np.ndarray, x_est: float | np.ndarray) -> Any:
    """
    Calculate mean relative absolute error.

    :param x: ground truth value
    :param x_est: estimated value
    :return: mean relative absolute error
    """
    return np.mean(abs((x - x_est) / x))
