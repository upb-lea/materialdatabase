"""Empirical functions."""
import numpy as np
import numpy.typing as npt


# ----------------
# Temperature Fits
# ----------------

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


# ----------------
# Permeability Fits
# ----------------

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


def enhanced_steinmetz(fb: tuple[float | np.ndarray, float | np.ndarray],
                       alpha: float, beta: float, k: float,
                       k_b: float, k_f: float, k_alpha2: float) -> float | np.ndarray:
    """
    Enhanced Steinmetz loss model incorporating additional scaling terms dependent on frequency and flux density.

    :param fb: Tuple (f, B) of frequency and magnetic flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k: Base scaling factor
    :param k_b: Coefficient for flux density-dependent scaling
    :param k_f: Coefficient for frequency-dependent scaling
    :param k_alpha2: Exponent for frequency in additional scaling term
    :return: Power loss density
    """
    f, b = fb
    return (k + k_b * b + k_f * f ** k_alpha2) * f ** alpha * b ** beta


def enhanced_steinmetz_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                          alpha: float, beta: float,
                          k_b: float, k_f: float, k_alpha2: float,
                          c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Temperature-dependent enhanced Steinmetz model using quadratic temperature function and additional scaling terms.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k_b: Coefficient for flux density-dependent scaling
    :param k_f: Coefficient for frequency-dependent scaling
    :param k_alpha2: Exponent for frequency in additional scaling term
    :param c_0: Constant coefficient for temperature scaling
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Power loss density
    """
    f, T, b = fTb
    k = quadratic_temperature(T, c_0, c_1, c_2)
    return (k + k_b * b + k_f * f ** k_alpha2) * f ** alpha * b ** beta


def log_enhanced_steinmetz_qT(fTb: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
                              alpha: float, beta: float,
                              k_b: float, k_f: float, k_alpha2: float,
                              c_0: float, c_1: float, c_2: float) -> float | np.ndarray:
    """
    Logarithm of the temperature-dependent enhanced Steinmetz model with frequency and flux-density dependent scaling.

    :param fTb: Tuple (f, T, B) of frequency, temperature, and flux density
    :param alpha: Frequency exponent
    :param beta: Flux density exponent
    :param k_b: Coefficient for flux density-dependent scaling
    :param k_f: Coefficient for frequency-dependent scaling
    :param k_alpha2: Exponent for frequency in additional scaling term
    :param c_0: Constant coefficient for temperature scaling
    :param c_1: Linear temperature coefficient
    :param c_2: Quadratic temperature coefficient
    :return: Logarithm of power loss density
    """
    return np.log(enhanced_steinmetz_qT(fTb, alpha, beta, k_b, k_f, k_alpha2, c_0, c_1, c_2))


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
    """
    Fit function for amplitude permeability μₐ(B, T) based on polynomial B-dependence and linear temperature scaling.

    Typically accurate for b < 0.3 T.

    :param _Tb:
    :param mur_0: Base relative permeability (T-independent offset)
    :param mur_1: Polynomial coefficients B^1
    :param mur_2: Polynomial coefficients B^2
    :param mur_3: Polynomial coefficients B^3
    :param mur_4: Polynomial coefficients B^4
    :param c_0: Temperature scaling coefficient for constant offset term
    :param c_1: Temperature scaling coefficient for B-dependent terms
    :return: Amplitude permeability μₐ
    """
    _, T, b = _Tb

    k_0 = 1 + T * c_0  # Temperature scaling for base permeability
    k_1 = 1 + T * c_1  # Temperature scaling for B-dependent polynomial

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
    """
    Polynomial (up to B^2) and logarithmic frequency fit for amplitude permeability μₐ(f, B, T).

    Parameters:
        fTb: Tuple (f, T, B)
            f (float or np.ndarray): Frequency in Hz
            T (float or np.ndarray): Temperature in °C
            B (float or np.ndarray): Magnetic flux density in T
        mur_0: Base permeability (T-dependent only)
        mur_1: Linear B coefficient
        mur_2: Quadratic B coefficient
        c_0: Temperature coefficient for base permeability
        c_1: Temperature coefficient for B-dependent terms
        c_f: Frequency coefficient (log-scale)

    Returns:
        float or np.ndarray: Fitted amplitude permeability μₐ
    """
    f, T, B = fTb

    # Temperature scaling
    k_0 = 1 + T * c_0
    k_1 = 1 + T * c_1

    # Frequency influence (linear)
    k_f = 1 + c_f * f

    return (mur_0 * k_0 + k_1 * (mur_1 * B + mur_2 * B ** 2)) * k_f


# ----------------
# Permittivity (Conductivity) Fits
# ----------------
def fit_sigma_fT(fT: tuple[float | np.ndarray, float | np.ndarray],
                 c_0: float,
                 c_T1: float, c_T2: float,
                 c_f1: float, c_f2: float, c_f3: float,
                 c_mix11: float, c_mix21: float, c_mix12: float, c_mix31: float,
                 ) -> float | np.ndarray:
    """
    Temperature-dependent polynomial fit suitable for conductivity fit. (-> used as permittivity fit)

    :param fT: tuple of frequency and temperature
    :param c_0: constant coefficient
    :param c_T1: linear temperature coefficient
    :param c_T2: quadratic temperature coefficient
    :param c_f1: linear frequency coefficient
    :param c_f2: quadratic frequency coefficient
    :param c_f3: cubic frequency coefficient
    :param c_mix11: deg(f)=1, deg(T)=1
    :param c_mix21: deg(f)=2, deg(T)=1
    :param c_mix12: deg(f)=1, deg(T)=2
    :param c_mix31: deg(f)=3, deg(T)=1
    :return: fitted permittivity (real, imaginary, amplitude or loss angle)
    """
    f, T = fT
    return (c_0 +  # bulk conductivity
            c_T1 * T + c_T2 * T ** 2 +  # temperature coefficients -> modify bulk conductivity
            c_f1 * f + c_f2 * f ** 2 + c_f3 * f ** 3 +  # frequency coefficients -> modify ac conductivity
            c_mix11 * f * T + c_mix21 * f ** 2 * T + c_mix12 * f * T ** 2 + c_mix31 * f ** 3 * T)  # mix terms
