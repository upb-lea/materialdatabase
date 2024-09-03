"""Collection of smaller functions."""
import copy
import numpy as np
import ntpath

j = complex(0, 1)


def crop(x_full: np.ndarray, y_full: np.ndarray, xa: float, xb: float):
    """
    Crops two 1D arrays for the interval [a, b] according to x without changing the inputs.

    :param x_full: array of the uncropped x-data
    :type x_full: ndarray
    :param y_full: array of the uncropped y-data
    :type y_full: ndarray
    :param xa: lower boundary of the interval
    :type xa: float
    :param xb: upper boundary of the interval
    :Type xb: float
    :return: croped arrays of x_full and y_full
    """
    x = copy.deepcopy(x_full)
    y = copy.deepcopy(y_full)
    y = y[x <= xb]
    x = x[x <= xb]
    y = y[xa <= x]
    x = x[xa <= x]
    return x, y


def path_leaf(path: str):
    """
    Return the tail of the path.

    :param path: path
    :type path: str
    :return: tail of path
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def L_from_Z(Z: np.ndarray | float, phi_deg: np.ndarray | float, f: np.ndarray | float):
    """
    Calculate the inductance L based on amplitude and phase angle of the impedance Z.

    :param Z: measured impedance Z in Ω
    :type Z: ndarray or float
    :param phi_deg: measured phase angle phi in degree
    :type phi_deg: ndarray or float
    :param f: measured frequency f in Hz
    :type f: ndarray or float
    :return: calculated inductance in H based on given data
    """
    return Z * np.array(np.sin(np.deg2rad(phi_deg)) - j * np.cos(np.deg2rad(phi_deg))) / (2*np.pi*f)


def get_closest(to_search: list | np.ndarray, x: list | np.ndarray):
    """
    Search an array for specific values and return the closest values.

    :param to_search: array with values to search for
    :type to_search: ndarray or list
    :param x: array to search for the values
    :type x: ndarray or list
    :return: list with the closest values to the wanted values
    """
    neighbours = []
    for element in to_search:
        index = 0
        for index, val in enumerate(x):
            if val >= element:
                if abs(element - x[index-1]) < abs(element - val):
                    neighbours.append(index-1)
                else:
                    neighbours.append(index - 1)
                break
    return neighbours


def Z_from_amplitude_and_angle(amplitude: float | np.ndarray, angle_deg: float | np.ndarray):
    """
    Calculate the complex impedance given the amplitude and phase angle of the impedance.

    :param amplitude: amplitude of the impedance in Ω
    :type amplitude: ndarray or float
    :param angle_deg: phase angle of the impedance in degree
    :type angle_deg: ndarray or float
    :return: complex impedance in Ω
    """
    return amplitude * np.array(np.cos(np.deg2rad(angle_deg)) + j * np.sin(np.deg2rad(angle_deg)))
