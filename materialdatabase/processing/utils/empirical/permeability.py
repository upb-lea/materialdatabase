"""Empirical permeability and magnetic-loss models."""

import numpy as np
import numpy.typing as npt

from materialdatabase.processing.utils.empirical.temperature import quadratic_temperature


def steinmetz(fb: tuple[float | np.ndarray, float | np.ndarray],
              alpha: float, beta: float, k: float | np.ndarray) -> float | np.ndarray:
    """
    Classic Steinmetz loss model: P proportional to f^alpha * B^beta with scaling factor k.

    :param fb: Tuple (f, B) of frequency and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k: Scaling factor (can be temperature-dependent)
    :return: Power loss density
    """
    f, b = fb
    return k * f ** alpha * b ** beta


def steinmetz_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                 alpha: float, beta: float, c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """Temperature-dependent Steinmetz model using quadratic temperature scaling."""
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return steinmetz((f, b), alpha, beta, k)


def enhanced_steinmetz(fb: tuple[float | np.ndarray, float | np.ndarray],
                       alpha: float, beta: float, k: float,
                       k_b: float, k_f: float, k_alpha2: float) -> float | np.ndarray:
    """Enhanced Steinmetz loss model with frequency and flux-density terms."""
    f, b = fb
    return (k + k_b * b + k_f * f ** k_alpha2) * f ** alpha * b ** beta


def enhanced_steinmetz_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                          alpha: float, beta: float,
                          k_b: float, k_f: float, k_alpha2: float,
                          c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """Temperature-dependent enhanced Steinmetz model."""
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return (k + k_b * b + k_f * f ** k_alpha2) * f ** alpha * b ** beta


def temperature_enhanced_steinmetz_qT(
        fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
        alpha: float, beta: float,
        k_b: float, k_f: float, k_alpha2: float,
        c_0: float, c_1: float, c_2: float
) -> float | np.ndarray:
    """Enhanced Steinmetz model with temperature-scaled frequency dependence."""
    f, T, b = fTb
    temperature_factor = c_0 - c_1 * T + c_2 * T ** 2
    return (temperature_factor + k_b * b + k_f * T * f ** k_alpha2) * f ** alpha * b ** beta


def fit_mu_abs_TDK_MDT(
        _Tb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
        mur_0: float,
        mur_1: float,
        mur_2: float,
        mur_3: float,
        mur_4: float,
        c_0: float,
        c_1: float
) -> float | npt.NDArray[np.float64]:
    """Fit amplitude permeability using a B-polynomial and temperature scaling."""
    _, T, b = _Tb

    k_0 = 1 + T * c_0
    k_1 = 1 + T * c_1

    return mur_0 * k_0 + k_1 * (mur_1 * b + mur_2 * b ** 2 + mur_3 * b ** 3 + mur_4 * b ** 4)


def fit_mu_abs_LEA_MTB_MagNet(
        fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
        mur_0: float,
        mur_1: float,
        mur_2: float,
        c_0: float,
        c_1: float,
        c_f: float
) -> float | np.ndarray:
    """Fit amplitude permeability using B, temperature, and frequency terms."""
    f, T, B = fTb

    k_0 = 1 + T * c_0
    k_1 = 1 + T * c_1
    k_f = 1 + c_f * f

    return (mur_0 * k_0 + k_1 * (mur_1 * B + mur_2 * B ** 2)) * k_f