# all static functions shall be inserted in this file

# Python integrated libraries
import math
import json
import os

# 3rd party libraries
import numpy as np
from scipy.interpolate import interp1d


# local libraries

# ------Remove Duplicate from freq array------
def remove(arr, n):
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


def store_data(material_name, data_to_be_stored):
    """
    Method is used to store data from measurement/datasheet into the material database.
    :param material_name:
    :param data_to_be_stored:
    :return:
    """
    with open('material_data_base.json', 'w') as outfile:
        json.dump(data_to_be_stored, outfile, indent=4)
    mdb_print(f"Material properties of {material_name} are stored in the material database.")


# -----find nearby frequency n Temp---------
def find_nearest(array, value):
    array = np.asarray(array)
    array.sort()
    idx = (np.abs(array - value)).argmin()
    if array[idx] > value:
        return array[idx - 1], array[idx]
    else:
        if idx == len(array) - 1:
            return array[idx - 1], array[idx]
        else:
            return array[idx], array[idx + 1]


def set_silent_status(is_silent: bool):
    """
    Silent mode global variable.

    :param is_silent: True for silent mode, False for mode with print outputs
    :type is_silent: bool
    """
    global silent
    silent = is_silent


def mdb_print(text: str, end='\n'):
    """
    Print function what checks the silent-mode-flag.
    Print only in case of no-silent-mode.

    :param text: Text to print
    :type text: str
    :param end: command for end of line, e.g. '\n' or '\t'
    :type end: str

    """
    if not silent:
        print(text, end)


def rect(r, theta):
    """theta in degrees

    returns tuple; (float, float); (x,y)
    """
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x, y


def find_nearest_neighbours(value, list_to_search_in):
    """
    only works for sorted lists (small to big)

    Case 0: if len(list_to_search_in) == 1: return duplicated
    Case 1: if value == any(list_to_search_in): return duplicated
    Case 2: if value inbetween: return neighbours
    Case 3:
        a) if value smaller than data: return smallest two
        b) if value is bigger than data: return biggest two

    :param value:
    :param list_to_search_in:
    :return:
    """
    if len(list_to_search_in) == 1:  # Case 0
        return 0, list_to_search_in[0], 0, list_to_search_in[0]
    else:
        value_low, value_high = 0, 0
        index_low, index_high = 0, 0
        if value < list_to_search_in[0]:  # Case 3a)
            return 0, list_to_search_in[0], 1, list_to_search_in[1]

        for index_data, value_data in enumerate(list_to_search_in):
            # print(value)
            # print(value_data)
            if value_data < value:
                value_low = value_data
                index_low = index_data
            if value_data == value:  # Case 1
                value_low, value_high = value, value
                index_low, index_high = index_data, index_data
                break
            if value_data > value:
                value_high = value_data
                index_high = index_data
                break

        if value_high == 0:  # Case 3b: check if input value is bigger than any value in list
            value_low = list_to_search_in[len(list_to_search_in) - 2]
            index_low = len(list_to_search_in) - 2
            value_high = list_to_search_in[len(list_to_search_in) - 1]
            index_high = len(list_to_search_in) - 1
        return index_low, value_low, index_high, value_high


def create_permittivity_neighbourhood(T, f, list_of_permittivity_dicts):
    """
    
    :param T: 
    :param f: 
    :param list_of_permittivity_dicts: 
    :return: 
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "T": {
                    "value": None,
                    "index": None
                },
                "f": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_low_f_high":
            {
                "T": {
                    "value": None,
                    "index": None
                },
                "f": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_high_f_low":
            {
                "T": {
                    "value": None,
                    "index": None
                },
                "f": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_high_f_high":
            {
                "T": {
                    "value": None,
                    "index": None
                },
                "f": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
    }

    # In permittivity data:
    # find two temperatures at which were measured that are closest to given T
    temperatures = []
    for permittivity_dict in list_of_permittivity_dicts:
        temperatures.append(permittivity_dict["temperature"])  # store them in a list
    index_T_low_neighbour, value_T_low_neighbour, index_T_high_neighbour, value_T_high_neighbour = \
        find_nearest_neighbours(T, temperatures)

    nbh["T_low_f_low"]["T"]["value"], nbh["T_low_f_high"]["T"]["value"] = value_T_low_neighbour, value_T_low_neighbour
    nbh["T_low_f_low"]["T"]["index"], nbh["T_low_f_high"]["T"]["index"] = index_T_low_neighbour, index_T_low_neighbour
    nbh["T_high_f_low"]["T"]["value"], nbh["T_high_f_high"]["T"]["value"] = value_T_high_neighbour, value_T_high_neighbour
    nbh["T_high_f_low"]["T"]["index"], nbh["T_high_f_high"]["T"]["index"] = index_T_high_neighbour, index_T_high_neighbour

    # T low
    nbh["T_low_f_low"]["f"]["index"], nbh["T_low_f_low"]["f"]["value"], \
    nbh["T_low_f_high"]["f"]["index"], nbh["T_low_f_high"]["f"]["value"] = \
        find_nearest_neighbours(f, list_of_permittivity_dicts[index_T_low_neighbour]["frequencies"])

    nbh["T_low_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["T"]["index"]]["epsilon_r"][nbh["T_low_f_low"]["f"]["index"]]
    nbh["T_low_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["T"]["index"]]["epsilon_phi_deg"][nbh["T_low_f_low"]["f"]["index"]]
    nbh["T_low_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["T"]["index"]]["epsilon_r"][nbh["T_low_f_high"]["f"]["index"]]
    nbh["T_low_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["T"]["index"]]["epsilon_phi_deg"][nbh["T_low_f_high"]["f"]["index"]]

    # T high
    nbh["T_high_f_low"]["f"]["index"], nbh["T_high_f_low"]["f"]["value"], \
    nbh["T_high_f_high"]["f"]["index"], nbh["T_high_f_high"]["f"]["value"] = \
        find_nearest_neighbours(f, list_of_permittivity_dicts[index_T_high_neighbour]["frequencies"])

    nbh["T_high_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["T"]["index"]]["epsilon_r"][nbh["T_high_f_low"]["f"]["index"]]
    nbh["T_high_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["T"]["index"]]["epsilon_phi_deg"][nbh["T_high_f_low"]["f"]["index"]]
    nbh["T_high_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["T"]["index"]]["epsilon_r"][nbh["T_high_f_high"]["f"]["index"]]
    nbh["T_high_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["T"]["index"]]["epsilon_phi_deg"][nbh["T_high_f_high"]["f"]["index"]]

    return nbh


def my_interpolate_linear(a, b, f_a, f_b, x):
    """
    interpolates linear for a < x < b
    :param a:
    :param b:
    :param f_a:
    :param f_b:
    :param x:
    :return:
    """
    slope = (f_b - f_a) / (b - a)
    f_x = slope * (x - a) + f_a
    return f_x


def my_polate_linear(a, b, f_a, f_b, x):
    """
    interpolates or extrapolates linear for a<x<b or x<a and x>b
    :param a:
    :param b:
    :param f_a:
    :param f_b:
    :param x:
    :return:
    """
    if a == b == x and f_a == f_b:
        f_x = f_a
    else:
        slope = (f_b - f_a) / (b - a)
        f_x = slope * (x - a) + f_a
    return f_x


def interpolate_neighbours_linear(T, f, neighbours):
    """

    :param T:
    :param f:
    :param neighbours:
    :return:
    """
    # Interpolation of Amplitude
    # in temperature at f_low
    epsilon_r_at_T_f_low = my_polate_linear(a=neighbours["T_low_f_low"]["T"]["value"], b=neighbours["T_high_f_low"]["T"]["value"],
                                            f_a=neighbours["T_low_f_low"]["epsilon_r"], f_b=neighbours["T_high_f_low"]["epsilon_r"],
                                            x=T)
    # in temperature at f_high
    epsilon_r_at_T_f_high = my_polate_linear(a=neighbours["T_low_f_high"]["T"]["value"], b=neighbours["T_high_f_high"]["T"]["value"],
                                             f_a=neighbours["T_low_f_high"]["epsilon_r"], f_b=neighbours["T_high_f_high"]["epsilon_r"],
                                             x=T)
    # between f_low and f_high
    epsilon_r = my_polate_linear(a=neighbours["T_low_f_low"]["f"]["value"], b=neighbours["T_low_f_high"]["f"]["value"],
                                 f_a=epsilon_r_at_T_f_low, f_b=epsilon_r_at_T_f_high, x=f)

    # Interpolation of Phase
    # in temperature at f_low
    epsilon_phi_deg_at_T_f_low = my_polate_linear(a=neighbours["T_low_f_low"]["T"]["value"], b=neighbours["T_high_f_low"]["T"]["value"],
                                                  f_a=neighbours["T_low_f_low"]["epsilon_phi_deg"], f_b=neighbours["T_high_f_low"]["epsilon_phi_deg"],
                                                  x=T)
    # in temperature at f_high
    epsilon_phi_deg_at_T_f_high = my_polate_linear(a=neighbours["T_low_f_high"]["T"]["value"], b=neighbours["T_high_f_high"]["T"]["value"],
                                                   f_a=neighbours["T_low_f_high"]["epsilon_phi_deg"], f_b=neighbours["T_high_f_high"]["epsilon_phi_deg"],
                                                   x=T)
    # between f_low and f_high
    epsilon_phi_deg = my_polate_linear(a=neighbours["T_low_f_low"]["f"]["value"], b=neighbours["T_low_f_high"]["f"]["value"],
                                       f_a=epsilon_phi_deg_at_T_f_low, f_b=epsilon_phi_deg_at_T_f_high, x=f)

    return epsilon_r, epsilon_phi_deg
