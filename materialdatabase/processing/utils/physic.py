"""Physical functions."""
import numpy as np
from materialdatabase.processing.utils.constants import epsilon_0

def pv_mag(f: float | np.ndarray, mu_imag: float | np.ndarray, h_abs: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the magentic loss density in frequency domain.

    :param f: frequency
    :param mu_imag: imaginary part of the permeability
    :param h_abs: absolut value of the magnetic field
    :return:
    """
    return -0.5 * 2 * np.pi * f * mu_imag * h_abs ** 2


def pv_el(f: float | np.ndarray, eps_imag: float | np.ndarray, e_abs: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the (di-)electric loss density in frequency domain.

    :param f: frequency
    :param eps_imag: imaginary part of the permeability
    :param e_abs: absolut value of the magnetic field
    :return:
    """
    return -0.5 * 2 * np.pi * f * eps_imag * e_abs ** 2


def mu_imag_from_pv(f: float | np.ndarray, h_abs: float | np.ndarray, p_hyst: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the imaginary part of the permeability for a given loss density in frequency domain.

    :param f: frequency
    :param p_hyst: (hysteresis) loss density
    :param h_abs: absolut value of the magnetic field
    :return: imaginary part of the permeability
    """
    return p_hyst / (-0.5 * 2 * np.pi * f * h_abs ** 2)

def eps_r_from_sigma(f, sigma):
    """
    Calculate the (complex) relative permittivity from a (complex) conductivity.

    :param f: frequency
    :param sigma: (complex) conductivity
    :return: (complex) relative permittivity
    """
    return sigma / (1j * 2 * np.pi * f * epsilon_0)

def sigma_from_eps_r(f, eps_r):
    """
    Calculate the (complex) conductivity from a (complex) relative permittivity.

    :param f: frequency
    :param eps_r: (complex) relative permittivity
    :return: (complex) relative conductivity
    """
    return 1j * 2 * np.pi * f * eps_r * epsilon_0
