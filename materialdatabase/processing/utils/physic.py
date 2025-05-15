"""Physical functions."""
import numpy as np


def pv_mag(f: float, mu_imag: float, h_abs: float) -> float:
    """
    Calculate the magentic loss density in frequency domain.

    :param f: frequency
    :param mu_imag: imaginary part of the permeability
    :param h_abs: absolut value of the magnetic field
    :return:
    """
    return -0.5 * 2 * np.pi * f * mu_imag * h_abs ** 2


def pv_el(f: float, eps_imag: float, e_abs: float) -> float:
    """
    Calculate the (di-)electric loss density in frequency domain.

    :param f: frequency
    :param eps_imag: imaginary part of the permeability
    :param e_abs: absolut value of the magnetic field
    :return:
    """
    return -0.5 * 2 * np.pi * f * eps_imag * e_abs ** 2
