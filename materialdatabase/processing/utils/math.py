"""Mathmatical functions."""
import numpy as np

def mre(x, x_est):
    return np.mean(abs((x - x_est) / x))
