"""Empirical conductivity and permittivity models."""

import numpy as np


def fit_sigma_fT(fT: tuple[float | np.ndarray, float | np.ndarray],
                 c_0: float,
                 c_T1: float, c_T2: float,
                 c_f1: float, c_f2: float, c_f3: float,
                 c_mix11: float, c_mix21: float, c_mix12: float, c_mix31: float,
                 ) -> float | np.ndarray:
    """Temperature-dependent polynomial fit suitable for conductivity data."""
    f, T = fT

    c_T = c_T1 * T + c_T2 * T ** 2
    c_f = c_f1 * f + c_f2 * f ** 2 + c_f3 * f ** 3
    c_mix = c_mix11 * f * T + c_mix21 * f ** 2 * T + c_mix12 * f * T ** 2 + c_mix31 * f ** 3 * T

    return c_0 + c_T + c_f + c_mix
