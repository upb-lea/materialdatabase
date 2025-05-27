"""Empirical functions."""
import numpy as np


def quadratic_temperature(T: float | np.ndarray, c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Quadratic temperature dependence: k(T) = c₀ - c₁·T + c₂·T².

    :param T: Temperature in °C or K
    :param c_0: Constant coefficient
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Temperature-dependent scaling factor
    """
    return c_0 - c_1 * T + c_2 * T ** 2


def steinmetz(fb: tuple[float | np.ndarray, float | np.ndarray],
              alpha: float, beta: float, k: float | np.ndarray) -> float | np.ndarray:
    """
    Classic Steinmetz loss model: P ∝ f^α · B^β with scaling factor k.

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
    """
    Temperature-dependent Steinmetz model using quadratic temperature function.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param c_0: Constant coefficient for temperature scaling
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Power loss density
    """
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return steinmetz((f, b), alpha, beta, k)


def log_steinmetz_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                 alpha: float, beta: float, c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Temperature-dependent Steinmetz model using quadratic temperature function.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param c_0: Constant coefficient for temperature scaling
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Power loss density
    """
    return np.log(steinmetz_qT(fTb, alpha, beta, c_0, c_1, c_2))


def fit_mu_abs(fb: tuple[float | np.ndarray, float | np.ndarray],
               A: float, beta: float, b0: float, C: float, f0: float, n: float) -> float | np.ndarray:
    """
    Fit for amplitude permeability μₐ(B, f) using a Gaussian in B and decay in f.

    :param fb: Tuple (f, B) of frequency and flux density
    :param A: Amplitude of Gaussian peak
    :param beta: Controls the width of the Gaussian
    :param b0: Center of the Gaussian (optimal B)
    :param C: Offset or baseline permeability
    :param f0: Characteristic frequency of decay
    :param n: Order of frequency decay
    :return: Amplitude permeability μₐ
    """
    f, b = fb
    gauss_b = A * np.exp(-beta * (b - b0) ** 2) + C
    decay_f = 1 / (1 + (f / f0) ** n)
    return gauss_b * decay_f


def fit_mu_abs_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                  A: float, beta: float, b0: float, C: float, f0: float, n: float,
                  c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Fit for amplitude permeability μₐ(B, f, T) with temperature scaling factor.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and flux density
    :param A: Amplitude of Gaussian peak
    :param beta: Controls the width of the Gaussian
    :param b0: Center of the Gaussian (optimal B)
    :param C: Offset or baseline permeability
    :param f0: Characteristic frequency of decay
    :param n: Order of frequency decay
    :param c_0: Constant coefficient for temperature scaling
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Amplitude permeability μₐ
    """
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return fit_mu_abs((f, b), A, beta, b0, C, f0, n) * k
