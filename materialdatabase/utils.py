"""Collection of smaller functions."""
import copy
import numpy as np
import ntpath

j = complex(0, 1)


def crop(x_full, y_full, xa, xb):
    """
    Crops two 1D arrays for the interval [a, b] according to x without changing the inputs.

    :param x_full: array of the uncropped x-data
    :param y_full: array of the uncropped y-data
    :param xa: lower boundary of the interval
    :param xb: upper boundary of the interval
    :return: croped arrays of x_full and y_full
    """
    x = copy.deepcopy(x_full)
    y = copy.deepcopy(y_full)
    y = y[x <= xb]
    x = x[x <= xb]
    y = y[xa <= x]
    x = x[xa <= x]
    return x, y


def path_leaf(path):
    """
    Return the tail of the path.

    :param path: path
    :return: tail of path
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def L_from_Z(Z, phi_deg, f):
    """
    Calculate the inductance L based on amplitude and phase angle of the impedance Z.

    :param Z: measured impedance Z in Ω
    :param phi_deg: measured phase angle phi in degree
    :param f: measured frequency f in Hz
    :return: calculated inductance in H based on given data
    """
    return Z * np.array(np.sin(np.deg2rad(phi_deg)) - j * np.cos(np.deg2rad(phi_deg))) / (2*np.pi*f)


def get_closest(to_search, x):
    """
    Search an array for specific values and return the closest values.

    :param to_search: array with values to search for
    :param x: array to search for the values
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


def Z_from_amplitude_and_angle(amplitude, angle_deg):
    """
    Calculate the complex impedance given the amplitude and phase angle of the impedance.

    :param amplitude: amplitude of the impedance in Ω
    :param angle_deg: phase angle of the impedance in degree
    :return: complex impedance in Ω
    """
    return amplitude * np.array(np.cos(np.deg2rad(angle_deg)) + j * np.sin(np.deg2rad(angle_deg)))
