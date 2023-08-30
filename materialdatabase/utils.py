import copy
import numpy as np
import ntpath

j = complex(0, 1)

def crop(x_full, y_full, xa, xb):
    """Crops two 1D arrays for the interval [a, b] according to x wothout changing the inputs.
    """
    x = copy.deepcopy(x_full)
    y = copy.deepcopy(y_full)
    y = y[x <= xb]
    x = x[x <= xb]
    y = y[xa <= x]
    x = x[xa <= x]
    return x, y

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def L_from_Z(Z, phi_deg, f):
    return Z * np.array(np.sin(np.deg2rad(phi_deg)) - j * np.cos(np.deg2rad(phi_deg))) / (2*np.pi*f)

def get_closest(to_search, x):
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
    return amplitude * np.array(np.cos(np.deg2rad(angle_deg)) + j * np.sin(np.deg2rad(angle_deg)))
