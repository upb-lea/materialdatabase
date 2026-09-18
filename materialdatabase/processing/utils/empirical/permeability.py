"""Empirical permeability and magnetic-loss models."""

import numpy as np
import numpy.typing as npt

from materialdatabase.processing.utils.empirical.temperature import (
    quadratic_temperature,
)


def steinmetz(
    fb: tuple[float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    k: float | np.ndarray,
) -> float | np.ndarray:
    """
    Classic Steinmetz loss model: P proportional to f^alpha * B^beta with scaling factor k.

    :param fb: Tuple (f, B) of frequency and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k: Scaling factor (can be temperature-dependent)
    :return: Power loss density
    """
    f, b = fb
    return k * f**alpha * b**beta


def steinmetz_qT(
    fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    c_0: float,
    c_1: float,
    c_2: float,
) -> float | np.ndarray:
    """
    Temperature-dependent Steinmetz model using quadratic temperature scaling.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :param c_2: Coefficient for the quadratic term in the temperature scaling
    :return: Power loss density
    """
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return steinmetz((f, b), alpha, beta, k)


def enhanced_steinmetz(
    fb: tuple[float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    k: float,
    k_b: float,
    k_f: float,
    k_alpha2: float,
) -> float | np.ndarray:
    """
    Enhanced Steinmetz loss model with frequency and flux-density terms.

    :param fb: Tuple (f, B) of frequency and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k: Base scaling factor
    :param k_b: Flux density scaling factor
    :param k_f: Frequency scaling factor
    :param k_alpha2: Exponent for the frequency term
    :return: Power loss density
    """
    f, b = fb
    return (k + k_b * b + k_f * f**k_alpha2) * f**alpha * b**beta


def enhanced_steinmetz_qT(
    fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    k_b: float,
    k_f: float,
    k_alpha2: float,
    c_0: float,
    c_1: float,
    c_2: float,
) -> float | np.ndarray:
    """
    Temperature-dependent enhanced Steinmetz model.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k_b: Flux density scaling factor
    :param k_f: Frequency scaling factor
    :param k_alpha2: Exponent for the frequency term
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :param c_2: Coefficient for the quadratic term in the temperature scaling
    :return: Power loss density
    """
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return (k + k_b * b + k_f * f**k_alpha2) * f**alpha * b**beta


def temperature_enhanced_steinmetz_qT(
    fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    k_b: float,
    k_f: float,
    k_alpha2: float,
    c_0: float,
    c_1: float,
    c_2: float,
) -> float | np.ndarray:
    """
    Enhanced Steinmetz model with temperature-scaled frequency dependence.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k_b: Flux density scaling factor
    :param k_f: Frequency scaling factor
    :param k_alpha2: Exponent for the frequency term
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :param c_2: Coefficient for the quadratic term in the temperature scaling
    :return: Power loss density
    """
    f, T, b = fTb
    norm_f = 100_000
    temperature_factor = c_0 - c_1 * T + c_2 * T**2
    return (
        (temperature_factor + k_b * b + k_f * T * (f / norm_f) ** k_alpha2)
        * f**alpha
        * b**beta
    )


def te_steinmetz(
    fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    alpha: float,
    beta: float,
    k_b: float,
    k_f: float,
    k_alpha2: float,
    c_0: float,
    c_1: float,
    c_2: float,
) -> float | np.ndarray:
    """
    Enhanced Steinmetz model with temperature-scaled frequency and flux density dependence.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k_b: Flux density scaling factor
    :param k_f: Frequency scaling factor
    :param k_alpha2: Exponent for the frequency term
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :param c_2: Coefficient for the quadratic term in the temperature scaling
    :return: Power loss density
    """
    f, T, b = fTb
    norm_f = 100_000
    c_T = c_0 - c_1 * T + c_2 * T**2
    return (c_T + k_b * b * T + k_f * T * (f / norm_f) ** k_alpha2) * f**alpha * b**beta


def fit_mu_abs_TDK_MDT(
    _Tb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    mur_0: float,
    mur_1: float,
    mur_2: float,
    mur_3: float,
    mur_4: float,
    c_0: float,
    c_1: float,
) -> float | npt.NDArray[np.float64]:
    """
    Fit amplitude permeability using a B-polynomial and temperature scaling.

    :param _Tb: Tuple (T, B) of temperature and magnetic flux density
    :param mur_0: Base permeability
    :param mur_1: First-order permeability coefficient
    :param mur_2: Second-order permeability coefficient
    :param mur_3: Third-order permeability coefficient
    :param mur_4: Fourth-order permeability coefficient
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :return: Amplitude permeability
    """
    _, T, b = _Tb

    k_0 = 1 + T * c_0
    k_1 = 1 + T * c_1

    return mur_0 * k_0 + k_1 * (mur_1 * b + mur_2 * b**2 + mur_3 * b**3 + mur_4 * b**4)


def fit_mu_abs_LEA_MTB_MagNet(
    fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    mur_0: float,
    mur_1: float,
    mur_2: float,
    c_0: float,
    c_1: float,
    c_f: float,
) -> float | np.ndarray:
    """
    Fit amplitude permeability using B, temperature, and frequency terms.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and magnetic flux density
    :param mur_0: Base permeability
    :param mur_1: First-order permeability coefficient
    :param mur_2: Second-order permeability coefficient
    :param c_0: Coefficient for the constant term in the temperature scaling
    :param c_1: Coefficient for the linear term in the temperature scaling
    :param c_f: Coefficient for the frequency term in the frequency scaling
    :return: Amplitude permeability
    """
    f, T, B = fTb

    k_0 = 1 + T * c_0
    k_1 = 1 + T * c_1
    k_f = 1 + c_f * f

    return (mur_0 * k_0 + k_1 * (mur_1 * B + mur_2 * B**2)) * k_f
