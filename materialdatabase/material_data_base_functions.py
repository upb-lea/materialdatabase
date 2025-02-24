"""Functions of the material database."""
# all static functions shall be inserted in this file

# Python integrated libraries
import json
# 3rd party libraries
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as savgol

# local libraries
from materialdatabase.constants import *
from materialdatabase.enumerations import *

# Relative path to the database json file
global relative_path_to_db
relative_path_to_db = "../data/material_data_base.json"


# Auxiliary functions ------------------------------------------------------------------------------------------------------------------------------------------
j = complex(0, 1)


def read_in_digitized_datasheet_plot(path: str):
    """
    Read in a csv-file containing the x- and y-data of a digitized plot from a manufacturer datasheet.

    Information regarding Digitization:
        - used program: WebPlotDigitizer (made by Ankit Rohatgi)
        - delta_x = 8 Px
        - delta_y = 8 Px
        - format to save: Sort by: X
                          Order: Ascending
                          Digits: 5 Fixed
                          Column Separator: ;

    :param path: path to csv-file
    :type path: str
    :return: list containing two lists with x- and y-data ([["x-data"], ["y-data"]])
    """
    data = np.genfromtxt(path, delimiter=";", dtype=str)

    data = list(zip(*data))
    x_values = [float(value.replace(",", ".")) for value in data[0]]
    y_values = [float(value.replace(",", ".")) for value in data[1]]
    data = [x_values, y_values]
    return data


def remove(arr: np.ndarray, n: int):
    """
    Remove duplicates from array.

    :param arr: array with duplicates
    :type arr: ndarray
    :param n: has no effect of the functionality
    :type n: int
    :return: array without duplicates
    """
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


def crop_data_fixed(x: list, pre_cropped_values: int = 0, post_cropped_values: int = 0):
    """
    Crop an array based on the given indices.

    IMPORTANT! THE SECOND INDEX IS COUNTED BACKWARDS(NEGATIVE)!

    :param x: array to get cropped
    :type x: list
    :param pre_cropped_values: start value
    :type pre_cropped_values: int
    :param post_cropped_values: end value, but counted backwards
    :type post_cropped_values: int
    :return: cropped data
    """
    if post_cropped_values == 0:
        post_cropped_values = -len(x)
    return x[pre_cropped_values:-post_cropped_values]


def crop_3_with_1(x: np.ndarray | list, y: np.ndarray | list, z: np.ndarray | list, xa: int | float, xb: int | float):
    """
    Crop three arrays based on one array.

    :param x: array crop is based on
    :type x: ndarray or list
    :param y: first array to get cropped
    :type y: ndarray or list
    :param z: second array to get cropped
    :type z: ndarray or list
    :param xa: start value of crop
    :type xa: float or int
    :param xb: end value of crop
    :type xb: float or int
    :return: the three cropped arrays
    """
    x_copy = np.array(x)
    y_copy = np.array(y)
    z_copy = np.array(z)
    filter_bool = [True] * x_copy.shape[0]

    statements = [list(x_copy > xa),
                  list(x_copy < xb)]

    for statement in statements:
        filter_bool = [a and zr for a, zr in zip(filter_bool, statement)]

    return x_copy[filter_bool], y_copy[filter_bool], z_copy[filter_bool]


def store_data(material_name: str, data_to_be_stored: dict) -> None:
    """
    Store data from measurement/datasheet into the material database.

    :param material_name: name of the material
    :type material_name: str
    :param data_to_be_stored: data to be stored
    :type data_to_be_stored: dict
    """
    with open('material_data_base.json', 'w') as outfile:
        json.dump(data_to_be_stored, outfile, indent=4)
    print(f"Material properties of {material_name} are stored in the material database.")


def load_material_from_db(material_name: str) -> None:
    """
    Load data from material database.

    :param material_name: name of material
    :type material_name: str
    :return: all data of specific material
    :rtype: dict
    """
    with open(relative_path_to_db, 'r') as database:
        print("Read data from the data base.")
        data_dict = json.load(database)
    return data_dict[material_name]


def find_nearest(array: np.ndarray | list, value: float):
    """
    Find the nearest value in an array.

    :param array: array to search
    :type array: ndarray or list
    :param value: desired value
    :type value: float
    :return: two values of the array with the wanted value in between
    """
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


def rect(radius_or_amplitude: np.ndarray | float, theta_deg: np.ndarray | float):
    """
    Convert polar coordinates [radius, angle] into cartesian coordinates [abscissa_x,ordinate_y].

    :param radius_or_amplitude: radius or amplitude
    :type radius_or_amplitude: ndarray or float
    :param theta_deg: angle in degree
    :type theta_deg: ndarray or float
    :return: abscissa_x, ordinate_y
    """
    abscissa_x = radius_or_amplitude * np.cos(np.radians(theta_deg))
    ordinate_y = radius_or_amplitude * np.sin(np.radians(theta_deg))
    return abscissa_x, ordinate_y


def sort_data(a: np.ndarray | list, b: np.ndarray | list, c: np.ndarray | list):
    """
    Sort three arrays according to array a.

    :param a: array that is the base of the sorting
    :type a: ndarray or list
    :param b: array that gets sorted based on a
    :type b: ndarray or list
    :param c: array that gets sorted based on a
    :type c: ndarray or list
    :return: the three arrays sorted according to a
    """
    sorted_list_of_lists = [list(x) for x in list(zip(*sorted(zip(a, b, c), key=lambda a: a)))]
    return np.array(sorted_list_of_lists[0]), np.array(sorted_list_of_lists[1]), np.array(sorted_list_of_lists[2])


def interpolate_a_b_c(a: np.ndarray | list, b: np.ndarray | list, c: np.ndarray | list, no_interpolation_values: int = 20):
    """
    Interpolation between three arrays based on the first array.

    :param a: array that is the base of the interpolation
    :type a: ndarray or list
    :param b: array that gets interpolated based on the values of array a
    :type b: ndarray or list
    :param c: array that gets interpolated based on the values of array a
    :type c: ndarray or list
    :param no_interpolation_values: number of interpolation values
    :type no_interpolation_values: int
    :return: the three interpolated arrays
    """
    # Find the border of the common magnetic flux density values
    b_min = min(a)
    b_max = max(a)
    # mdb_print(f"{b_max_min, b_min_max = }")
    # Create the magnetic flux density vector that is used for later interpolation actions

    a_reduced = np.linspace(b_min, b_max, no_interpolation_values)

    f_b_interpol = interp1d(a, b)
    f_c_interpol = interp1d(a, c)
    b_interpol_common = f_b_interpol(a_reduced)
    c_interpol_common = f_c_interpol(a_reduced)
    return a_reduced, b_interpol_common, c_interpol_common
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load Permeability --------------------------------------------------------------------------------------------------------------------------------------------
def updates_x_ticks_for_graph(x_data: np.ndarray | list, y_data: np.ndarray | list, x_new: np.ndarray | list | float, kind: str = "linear"):
    """
    Update the x-values of the given (x_data,y_data)-dataset and returns y_new based on x_new.

    :param x_data: x-data given
    :type x_data: ndarray or list
    :param y_data: y-data given
    :type y_data: ndarray or list
    :param x_new: new x-values
    :type x_new: ndarray or list or float
    :param kind: kind of interpolation
    :type kind: str
    :return: y_new-data corresponding to the x_new-data
    """
    f_linear = interp1d(x_data, y_data, kind=kind, fill_value="extrapolate")
    return f_linear(x_new)


def check_input_permeability_data(datasource: str, material_name: str, temperature: float, frequency: float) -> None:
    """
    Check input permeability data for correct input parameters.

     * datasource must be 'measurements' or 'manufacturer_datasheet'
     * material_name, T, f must be different from None

    :param datasource: datasource as a string
    :type datasource: str
    :param material_name: material name as a string
    :type material_name: str
    :param temperature: temperature in °C
    :type temperature: float
    :param frequency: frequency in Hz
    :type frequency: float
    """
    if datasource != MaterialDataSource.Measurement and datasource != MaterialDataSource.ManufacturerDatasheet:
        raise Exception("'datasource' must be 'manufacturer_datasheet' or 'measurements'.")

    if material_name is None or temperature is None or frequency is None:
        raise Exception(f"Failure in selecting data from materialdatabase. {material_name=}, {temperature=}, {frequency=}.")


def getdata_datasheet(permeability: np.ndarray | list, variable: float, frequency: float, temperature_1: float, temperature_2: float):
    """
    Interpolation of permeability data between two temperatures at a constant frequency.

    Linear Interpolation between temperature_1 and temperature_2 to get a value for the temperature "variable".

    :param permeability: permeability data
    :type permeability: ndarray or list
    :param variable: desired temperature value in °C
    :type variable: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature_1: first temperature value in °C
    :type temperature_1: float
    :param temperature_2: second temperature value in °C
    :type temperature_2: float
    :return: magnetic flux density, real part of permeability and imaginary part of permeability
    """
    for k in range(len(permeability)):
        if permeability[k]["frequency"] == frequency and permeability[k]["temperature"] == temperature_1:
            b_1 = permeability[k]["flux_density"]
            mu_real_1 = permeability[k]["mu_r_real"]
            mu_imag_1 = permeability[k]["mu_r_imag"]
            t_mu_imag_1 = interp1d(b_1, mu_imag_1)
            t_mu_real_1 = interp1d(b_1, mu_real_1)
        if permeability[k]["frequency"] == frequency and permeability[k]["temperature"] == temperature_2:
            b_2 = permeability[k]["flux_density"]
            mu_real_2 = permeability[k]["mu_r_real"]
            mu_imag_2 = permeability[k]["mu_r_imag"]
            t_mu_imag_2 = interp1d(b_2, mu_imag_2)
            t_mu_real_2 = interp1d(b_2, mu_real_2)

    # --------linear interpolation at constant freq-------------
    mu_i = []
    mu_r = []

    for y in range(len(b_1)):
        mu_r.append(t_mu_real_1(b_1[y]) + (t_mu_real_2(b_1[y]) - t_mu_real_1(b_1[y])) / (temperature_2 - temperature_1) * (variable - temperature_1))
        mu_i.append(t_mu_imag_1(b_1[y]) + (t_mu_imag_2(b_1[y]) - t_mu_imag_1(b_1[y])) / (temperature_2 - temperature_1) * (variable - temperature_1))
    return b_1, mu_r, mu_i


def create_permeability_neighbourhood_datasheet(temperature: float, frequency: float, list_of_permeability_dicts: np.ndarray | list):
    """
    Create a neighbourhood for permeability data of a datasheet.

    :param temperature: temperature value in °C
    :type temperature: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param list_of_permeability_dicts: list of permeability data dicts
    :type list_of_permeability_dicts: ndarray or list
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low": {"index": None,
                        "temperature": None,
                        "frequency": None,
                        "flux_density": None,
                        "mu_r_real": None,
                        "mu_r_imag": None},
        "T_low_f_high": {"index": None,
                         "temperature": None,
                         "frequency": None,
                         "flux_density": None,
                         "mu_r_real": None,
                         "mu_r_imag": None},
        "T_high_f_low": {"index": None,
                         "temperature": None,
                         "frequency": None,
                         "flux_density": None,
                         "mu_r_real": None,
                         "mu_r_imag": None},
        "T_high_f_high": {"index": None,
                          "temperature": None,
                          "frequency": None,
                          "flux_density": None,
                          "mu_r_real": None,
                          "mu_r_imag": None}
    }

    # In permeability data: find values of nearest neighbours
    T_value_low, T_value_high, f_value_low, f_value_high = find_nearest_neighbour_values_permeability(list_of_permeability_dicts, temperature, frequency)

    nbh["T_low_f_low"]["temperature"], nbh["T_low_f_high"]["temperature"] = T_value_low, T_value_low
    nbh["T_high_f_low"]["temperature"], nbh["T_high_f_high"]["temperature"] = T_value_high, T_value_high
    nbh["T_low_f_low"]["frequency"], nbh["T_high_f_low"]["frequency"] = f_value_low, f_value_low
    nbh["T_low_f_high"]["frequency"], nbh["T_high_f_high"]["frequency"] = f_value_high, f_value_high

    # find the indices of the neighbours in the original unsorted data
    for k_original, permeability_set in enumerate(list_of_permeability_dicts):

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_low:
            nbh["T_low_f_low"]["index"] = k_original
            nbh["T_low_f_low"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_low_f_low"]["mu_r_real"] = permeability_set["mu_r_real"]
            nbh["T_low_f_low"]["mu_r_imag"] = permeability_set["mu_r_imag"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_low:
            nbh["T_high_f_low"]["index"] = k_original
            nbh["T_high_f_low"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_high_f_low"]["mu_r_real"] = permeability_set["mu_r_real"]
            nbh["T_high_f_low"]["mu_r_imag"] = permeability_set["mu_r_imag"]

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_high:
            nbh["T_low_f_high"]["index"] = k_original
            nbh["T_low_f_high"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_low_f_high"]["mu_r_real"] = permeability_set["mu_r_real"]
            nbh["T_low_f_high"]["mu_r_imag"] = permeability_set["mu_r_imag"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_high:
            nbh["T_high_f_high"]["index"] = k_original
            nbh["T_high_f_high"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_high_f_high"]["mu_r_real"] = permeability_set["mu_r_real"]
            nbh["T_high_f_high"]["mu_r_imag"] = permeability_set["mu_r_imag"]

    return nbh


def create_permeability_neighbourhood_measurement(temperature: float, frequency: float, list_of_permeability_dicts: np.ndarray | list):
    """
    Create a neighbourhood for permeability data of a measurement.

    :param temperature: temperature value in °C
    :type temperature: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param list_of_permeability_dicts: list of permeability dicts
    :type list_of_permeability_dicts: ndarray or list
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low": {"index": None,
                        "temperature": None,
                        "frequency": None,
                        "flux_density": None,
                        "mu_r_abs": None,
                        "mu_phi_deg": None},
        "T_low_f_high": {"index": None,
                         "temperature": None,
                         "frequency": None,
                         "flux_density": None,
                         "mu_r_abs": None,
                         "mu_phi_deg": None},
        "T_high_f_low": {"index": None,
                         "temperature": None,
                         "frequency": None,
                         "flux_density": None,
                         "mu_r_abs": None,
                         "mu_phi_deg": None},
        "T_high_f_high": {"index": None,
                          "temperature": None,
                          "frequency": None,
                          "flux_density": None,
                          "mu_r_abs": None,
                          "mu_phi_deg": None}
    }

    # In permeability data: find values of nearest neighbours
    T_value_low, T_value_high, f_value_low, f_value_high = find_nearest_neighbour_values_permeability(list_of_permeability_dicts, temperature, frequency)

    nbh["T_low_f_low"]["temperature"], nbh["T_low_f_high"]["temperature"] = T_value_low, T_value_low
    nbh["T_high_f_low"]["temperature"], nbh["T_high_f_high"]["temperature"] = T_value_high, T_value_high
    nbh["T_low_f_low"]["frequency"], nbh["T_high_f_low"]["frequency"] = f_value_low, f_value_low
    nbh["T_low_f_high"]["frequency"], nbh["T_high_f_high"]["frequency"] = f_value_high, f_value_high

    # find the indices of the neighbours in the original unsorted data
    for k_original, permeability_set in enumerate(list_of_permeability_dicts):

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_low:
            nbh["T_low_f_low"]["index"] = k_original
            nbh["T_low_f_low"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_low_f_low"]["mu_r_abs"] = permeability_set["mu_r_abs"]
            nbh["T_low_f_low"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_low:
            nbh["T_high_f_low"]["index"] = k_original
            nbh["T_high_f_low"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_high_f_low"]["mu_r_abs"] = permeability_set["mu_r_abs"]
            nbh["T_high_f_low"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_high:
            nbh["T_low_f_high"]["index"] = k_original
            nbh["T_low_f_high"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_low_f_high"]["mu_r_abs"] = permeability_set["mu_r_abs"]
            nbh["T_low_f_high"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_high:
            nbh["T_high_f_high"]["index"] = k_original
            nbh["T_high_f_high"]["flux_density"] = permeability_set["flux_density"]
            nbh["T_high_f_high"]["mu_r_abs"] = permeability_set["mu_r_abs"]
            nbh["T_high_f_high"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

    return nbh


def find_nearest_neighbour_values_permeability(permeability_data: np.ndarray | list, temperature: float, frequency: float):
    """
    Find the nearest temperature and frequency values for a given neighbourhood of permeability data.

    :param permeability_data: permeability data
    :type permeability_data: ndarray or list
    :param temperature: temperature value in °C
    :type temperature: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :return: lower temperature value in °C, higher temperature value in °C, lower frequency value in Hz, higher frequency value in Hz
    """
    temperatures = []
    frequencies = []
    for permeability_set in permeability_data:
        temperatures.append(permeability_set["temperature"])
        frequencies.append(permeability_set["frequency"])

    # use sorted data without duplicates to find neighbours of operating point
    temperatures_sorted_without_duplicates = sorted(set(temperatures))
    frequencies_sorted_without_duplicates = sorted(set(frequencies))

    T_index_sorted_low, T_value_low, T_index_sorted_high, T_value_high = find_nearest_neighbours(temperature, temperatures_sorted_without_duplicates)
    f_index_sorted_low, f_value_low, f_index_sorted_high, f_value_high = find_nearest_neighbours(frequency, frequencies_sorted_without_duplicates)

    return T_value_low, T_value_high, f_value_low, f_value_high


def interpolate_b_dependent_quantity_in_temperature_and_frequency(temperature: float, frequency: float,
                                                                  temperature_low: float, temperature_high: float,
                                                                  frequency_low: float, frequency_high: float,
                                                                  b_t_low_f_low: np.ndarray, f_b_T_low_f_low: np.ndarray,
                                                                  b_T_high_f_low: np.ndarray, f_b_T_high_f_low: np.ndarray,
                                                                  b_T_low_f_high: np.ndarray, f_b_T_low_f_high: np.ndarray,
                                                                  b_T_high_f_high: np.ndarray, f_b_T_high_f_high: np.ndarray,
                                                                  no_interpolation_values: int = 8, y_label: str = None, plot: bool = False):
    """
    Interpolate a magnet flux density dependent quantity in temperature and frequency.

    :param temperature: desired temperature in °C
    :type temperature: float
    :param frequency: desired frequency  in Hz
    :type frequency: float
    :param temperature_low: lower temperature value in °C
    :type temperature_low: float
    :param temperature_high: higher temperature value in °C
    :type temperature_high: float
    :param frequency_low: lower frequency value in Hz
    :type frequency_low: float
    :param frequency_high: higher frequency value in Hz
    :type frequency_high: float
    :param b_t_low_f_low: magnetic flux density at the lower temperature in °C and the lower frequency value in Hz
    :type b_t_low_f_low: ndarray
    :param f_b_T_low_f_low: function dependent of b at the lower temperature in °C and the lower frequency value in Hz
    :type f_b_T_low_f_low: ndarray
    :param b_T_high_f_low: magnetic flux density at the higher temperature in °C and the lower frequency value in Hz
    :type b_T_high_f_low: ndarray
    :param f_b_T_high_f_low: function dependent of b at the higher temperature in °C and the lower frequency value in Hz
    :type f_b_T_high_f_low: ndarray
    :param b_T_low_f_high: magnetic flux density at the lower temperature in °C and the higher frequency value in Hz
    :type b_T_low_f_high: ndarray
    :param f_b_T_low_f_high: function dependent of b at the lower temperature in °C and the higher frequency value in Hz
    :type f_b_T_low_f_high: ndarray
    :param b_T_high_f_high: magnetic flux density at the higher temperature in °C and the higher frequency value in Hz
    :type b_T_high_f_high: ndarray
    :param f_b_T_high_f_high: function dependent of b at the higher temperature in °C and the higher frequency value in Hz
    :type f_b_T_high_f_high: ndarray
    :param no_interpolation_values: number of interpolation values
    :type no_interpolation_values: int
    :param y_label: label of y-axes
    :type y_label: str
    :param plot: enable/disable plotting of data
    :type plot: bool
    :return: array of magnetic flux density, arrays of function dependent of b
    """
    if len(b_t_low_f_low) != len(f_b_T_low_f_low):
        raise ValueError(f"b_T_low_f_low and f_b_T_low_f_low must have the same lengths: \n"
                         f"is {len(b_t_low_f_low), len(f_b_T_low_f_low)}")

    # Interpolate functions of input data
    f_T_low_f_low_interpol = interp1d(b_t_low_f_low, f_b_T_low_f_low)
    f_T_high_f_low_interpol = interp1d(b_T_high_f_low, f_b_T_high_f_low)
    f_T_low_f_high_interpol = interp1d(b_T_low_f_high, f_b_T_low_f_high)
    f_T_high_f_high_interpol = interp1d(b_T_high_f_high, f_b_T_high_f_high)
    # mdb_print(f_T_low_f_low_interpol(1.5))

    # Find the border of the common magnetic flux density values
    b_max_min = max(min(b_t_low_f_low), min(b_T_high_f_low), min(b_T_low_f_high), min(b_T_high_f_high))
    b_min_max = min(max(b_t_low_f_low), max(b_T_high_f_low), max(b_T_low_f_high), max(b_T_high_f_high))
    # mdb_print(f"{b_max_min, b_min_max = }")

    # Create the magnetic flux density vector that is used for later interpolation actions
    b_common = np.linspace(b_max_min, b_min_max, no_interpolation_values)

    # Create the function values according to the common magnetic flux density vector
    f_T_low_f_low_common = f_T_low_f_low_interpol(b_common)
    f_T_high_f_low_common = f_T_high_f_low_interpol(b_common)
    f_T_low_f_high_common = f_T_low_f_high_interpol(b_common)
    f_T_high_f_high_common = f_T_high_f_high_interpol(b_common)

    # Convert to numpy arrays
    f_T_low_f_low_common = np.array(f_T_low_f_low_common)
    f_T_high_f_low_common = np.array(f_T_high_f_low_common)
    f_T_low_f_high_common = np.array(f_T_low_f_high_common)
    f_T_high_f_high_common = np.array(f_T_high_f_high_common)

    # Manual linear interpolation:

    # First interpolate in temperature:
    if temperature_high == temperature_low:
        f_T_f_low_common = f_T_low_f_low_common  # at f_low
        f_T_f_high_common = f_T_low_f_high_common  # at f_high
    else:
        f_T_f_low_common = f_T_low_f_low_common + (f_T_high_f_low_common - f_T_low_f_low_common) / \
            (temperature_high - temperature_low) * (temperature - temperature_low)  # at f_low
        f_T_f_high_common = f_T_low_f_high_common + (f_T_high_f_high_common - f_T_low_f_high_common) / \
            (temperature_high - temperature_low) * (temperature - temperature_low)  # at f_high

    # Second interpolate in frequency:
    # mdb_print(f"{f_high, f_low = }")
    if frequency_low == frequency_high:
        f_T_f_common = f_T_f_low_common
    else:
        f_T_f_common = f_T_f_low_common + (f_T_f_high_common - f_T_f_low_common) / (frequency_high - frequency_low) * (frequency - frequency_low)

    if plot:
        scale = 1000
        plt.plot(b_common * scale, f_T_low_f_low_common, linestyle='dashed', color="tab:blue",
                 label=r"$T_\mathregular{low}$" + f"={temperature_low} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_low_f_high_common, linestyle='dashed', color="tab:red",
                 label=r"$T_\mathregular{low}$" + f"={temperature_low} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")

        plt.plot(b_common * scale, f_T_high_f_low_common, linestyle='dotted', color="tab:blue",
                 label=r"$T_\mathregular{high}$" + f"={temperature_high} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_high_f_high_common, linestyle='dotted', color="tab:red",
                 label=r"$T_\mathregular{high}$" + f"={temperature_high} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")

        plt.plot(b_common * scale, f_T_f_low_common, color="tab:blue", label=r"$T$" + f"={temperature} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_f_high_common, color="tab:red", label=r"$T$" + f"={temperature} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")
        plt.plot(b_common * scale, f_T_f_common, color="tab:orange", label=r"$T$" + f"={temperature} and " + r"$f$" + f"={frequency}")
        plt.xlabel("amplitude of magnetic flux density in mT")
        plt.ylabel(f"{y_label}")
        plt.title("Interpolation in temperature and frequency")
        plt.legend()
        plt.grid()
        plt.show()

    return b_common, f_T_f_common


def process_permeability_data(b_ref_raw: np.ndarray | list, mu_r_raw: np.ndarray | list, mu_phi_deg_raw: np.ndarray | list,
                              b_min: float = 0.05, b_max: float = 0.3, smooth_data: bool = False, crop_data: bool = False,
                              plot_data: bool = False, ax=None, f: float = None, T: float = None):
    """
    Post-Processing of raw data of the permeability.

    Function can smooth, crop and plot the permeability data.

    :param b_ref_raw: raw data of the magnetic flux density
    :type b_ref_raw: ndarray or float
    :param mu_r_raw: raw data of the amplitude of the permeability
    :type mu_r_raw: ndarray or float
    :param mu_phi_deg_raw: raw data of the angle of the permeability
    :type mu_phi_deg_raw: ndarray or float
    :param b_min: min value of the magnetic flux density for cropping
    :type b_min: float
    :param b_max: max value of the magnetic flux density for cropping
    :type b_max: float
    :param smooth_data: enable/disable smoothing of data (savgol-filter)
    :type smooth_data: bool
    :param crop_data: enable/disable cropping of data
    :type crop_data: bool
    :param plot_data: enable/disable plotting of data
    :type plot_data: bool
    :param ax: axes for plot
    :type ax: matplotlib.axes
    :param f: frequency value in Hz
    :type f: float
    :param T: temperature value in °C
    :type T: float
    :return: magnetic flux density and amplitude and angle of permeability
    """
    if crop_data:
        b_ref, mu_r, mu_phi_deg = crop_3_with_1(b_ref_raw, mu_r_raw, mu_phi_deg_raw, b_min, b_max)
    else:
        b_ref, mu_r, mu_phi_deg = b_ref_raw, mu_r_raw, mu_phi_deg_raw

    if smooth_data:
        mu_r = savgol(x=mu_r, window_length=int(len(b_ref) / 1), polyorder=2)
        mu_phi_deg = savgol(x=mu_phi_deg, window_length=int(len(b_ref) / 1), polyorder=2)

    # Extrapolate until b = 0
    # use linear extrapolation
    b_diff = b_ref[1] - b_ref[0]
    gradient_mu_r = (mu_r[1] - mu_r[0]) / b_diff
    mu_r_0 = mu_r[0] - gradient_mu_r * b_ref[0]
    gradient_mu_phi_deg = (mu_phi_deg[1] - mu_phi_deg[0]) / b_diff
    mu_phi_deg_0 = mu_phi_deg[0] - gradient_mu_phi_deg * b_ref[0]

    b_ref = np.insert(b_ref, 0, 0)
    mu_r = np.insert(mu_r, 0, mu_r_0)
    mu_phi_deg = np.insert(mu_phi_deg, 0, mu_phi_deg_0)
    # Check for negative values and correct them to be zero (can occur for phi at b close to zero)
    mu_phi_deg[mu_phi_deg < 0] = 0

    if plot_data:
        fig, ax = plt.subplots(3)

        # Amplitude Plot
        ax[0].plot(b_ref_raw*1000, mu_r_raw, label="raw")
        ax[0].plot(1000*b_ref, mu_r, "-x", label=f"{f=}, {T=}")
        ax[0].grid(True)
        ax[0].set_ylabel("rel. permeability")

        # Phase Plot
        ax[1].plot(b_ref_raw*1000, mu_phi_deg_raw, label="raw")
        # if smooth_data:
        ax[1].plot(1000*b_ref, mu_phi_deg, "-x", label=f"{f=}, {T=}")
        ax[1].grid(True)
        # ax[1].legend()

        ax[1].set_ylabel("loss angle in deg")

        # Loss Plot
        loss_density = p_hyst__from_mu_r_and_mu_phi_deg(frequency=f, b_peak=b_ref, mu_r=mu_r, mu_phi_deg=mu_phi_deg)
        ax[2].loglog(1000*b_ref, loss_density/1000, "-x", label=f"{f=}, {T=}")
        ax[2].grid(which="both", ls="-")
        ax[2].set_xlabel("magnetic flux density in mT")
        ax[2].set_ylabel("loss density in kW/m^3")
        # ax[2].legend()
        plt.show()

    return b_ref, mu_r, mu_phi_deg
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load Permittivity --------------------------------------------------------------------------------------------------------------------------------------------
def find_nearest_neighbours(value: float, list_to_search_in: np.ndarray | list):
    """
    Return the two values with the wanted value in between and additional the indices of the corresponding values.

    Only works for sorted lists (small to big).

    Case 0: if len(list_to_search_in) == 1: return duplicated
    Case 1: if value == any(list_to_search_in): return duplicated
    Case 2: if value inbetween: return neighbours
    Case 3a: value smaller than data: return smallest two
    Case 3b: if value is bigger than data: return biggest two

    :param value: desired value
    :type value: float
    :param list_to_search_in: array to search for value
    :type list_to_search_in: ndarray or list
    :return: lower index, lower value, higher index, higher value
    """
    if isinstance(value, str):
        raise TypeError("value must be int or float or list")

    if len(list_to_search_in) == 1:  # Case 0
        return 0, list_to_search_in[0], 0, list_to_search_in[0]
    else:
        value_low, value_high = 0, 0
        index_low, index_high = 0, 0
        if value < list_to_search_in[0]:  # Case 3a)
            return 0, list_to_search_in[0], 1, list_to_search_in[1]

        for index_data, value_data in enumerate(list_to_search_in):
            # mdb_print(value)
            # mdb_print(value_data)
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


def create_permittivity_neighbourhood(temperature: float, frequency: float, list_of_permittivity_dicts: np.ndarray | list):
    """
    Create neighbourhood for permittivity data.

    :param temperature: temperature value in °C
    :type temperature: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param list_of_permittivity_dicts: list of permittivity data dicts
    :type list_of_permittivity_dicts: ndarray or list
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low": {"temperature": {"value": None,
                                        "index": None},
                        "frequency": {"value": None,
                                      "index": None},
                        "epsilon_r": None,
                        "epsilon_phi_deg": None},
        "T_low_f_high": {"temperature": {"value": None,
                                         "index": None},
                         "frequency": {"value": None,
                                       "index": None},
                         "epsilon_r": None,
                         "epsilon_phi_deg": None},
        "T_high_f_low": {"temperature": {"value": None,
                                         "index": None},
                         "frequency": {"value": None,
                                       "index": None},
                         "epsilon_r": None,
                         "epsilon_phi_deg": None},
        "T_high_f_high": {"temperature": {"value": None,
                                          "index": None},
                          "frequency": {"value": None,
                                        "index": None},
                          "epsilon_r": None,
                          "epsilon_phi_deg": None},
    }
    # In permittivity data:
    # find two temperatures at which were measured that are closest to given T
    temperatures = []
    for permittivity_dict in list_of_permittivity_dicts:
        temperatures.append(permittivity_dict["temperature"])  # store them in a list
    index_T_low_neighbour, value_T_low_neighbour, index_T_high_neighbour, value_T_high_neighbour = find_nearest_neighbours(temperature, temperatures)

    nbh["T_low_f_low"]["temperature"]["value"], nbh["T_low_f_high"]["temperature"]["value"] = value_T_low_neighbour, value_T_low_neighbour
    nbh["T_low_f_low"]["temperature"]["index"], nbh["T_low_f_high"]["temperature"]["index"] = index_T_low_neighbour, index_T_low_neighbour
    nbh["T_high_f_low"]["temperature"]["value"], nbh["T_high_f_high"]["temperature"]["value"] = value_T_high_neighbour, value_T_high_neighbour
    nbh["T_high_f_low"]["temperature"]["index"], nbh["T_high_f_high"]["temperature"]["index"] = index_T_high_neighbour, index_T_high_neighbour

    # T low
    nbh["T_low_f_low"]["frequency"]["index"], nbh["T_low_f_low"]["frequency"]["value"], \
        nbh["T_low_f_high"]["frequency"]["index"], nbh["T_low_f_high"]["frequency"]["value"] = \
        find_nearest_neighbours(frequency, list_of_permittivity_dicts[index_T_low_neighbour]["frequencies"])

    nbh["T_low_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["temperature"]["index"]][
        "epsilon_r"][nbh["T_low_f_low"]["frequency"]["index"]]
    nbh["T_low_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["temperature"]["index"]][
        "epsilon_phi_deg"][nbh["T_low_f_low"]["frequency"]["index"]]
    nbh["T_low_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["temperature"]["index"]][
        "epsilon_r"][nbh["T_low_f_high"]["frequency"]["index"]]
    nbh["T_low_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["temperature"]["index"]][
        "epsilon_phi_deg"][nbh["T_low_f_high"]["frequency"]["index"]]

    # T high
    nbh["T_high_f_low"]["frequency"]["index"], nbh["T_high_f_low"]["frequency"]["value"], \
        nbh["T_high_f_high"]["frequency"]["index"], nbh["T_high_f_high"]["frequency"]["value"] = \
        find_nearest_neighbours(frequency, list_of_permittivity_dicts[index_T_high_neighbour]["frequencies"])

    nbh["T_high_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["temperature"]["index"]][
        "epsilon_r"][nbh["T_high_f_low"]["frequency"]["index"]]
    nbh["T_high_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["temperature"]["index"]][
        "epsilon_phi_deg"][nbh["T_high_f_low"]["frequency"]["index"]]
    nbh["T_high_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["temperature"]["index"]][
        "epsilon_r"][nbh["T_high_f_high"]["frequency"]["index"]]
    nbh["T_high_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["temperature"]["index"]][
        "epsilon_phi_deg"][nbh["T_high_f_high"]["frequency"]["index"]]

    return nbh


def create_steinmetz_neighbourhood(temperature: float, list_of_steinmetz_dicts: np.ndarray | list):
    """
    Create neighbourhood for steinmetz data.

    :param temperature: temperature value in °C
    :type temperature: float
    :param list_of_steinmetz_dicts: list of steinmetz data dicts
    :type list_of_steinmetz_dicts: ndarray or list
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low": {"temperature": {"value": None,
                                  "index": None},
                  "k": None,
                  "alpha": None,
                  "beta": None},
        "T_high": {"temperature": {"value": None,
                                   "index": None},
                   "k": None,
                   "alpha": None,
                   "beta": None}
    }

    # In permittivity data:
    # find two temperatures at which were measured that are closest to given T
    temperatures = []
    for steinmetz_dict in list_of_steinmetz_dicts:
        temperatures.append(steinmetz_dict["temperature"])  # store them in a list
    index_T_low_neighbour, value_T_low_neighbour, index_T_high_neighbour, value_T_high_neighbour = find_nearest_neighbours(temperature, temperatures)

    nbh["T_low"]["temperature"]["value"], nbh["T_low"]["temperature"]["index"] = value_T_low_neighbour, index_T_low_neighbour
    nbh["T_high"]["temperature"]["value"], nbh["T_high"]["temperature"]["index"] = value_T_high_neighbour, index_T_high_neighbour

    return nbh


def my_interpolate_linear(a: float, b: float, f_a: float, f_b: float, x: float):
    """
    Interpolates linear between to points 'a' and 'b'.

    The return value is f_x in dependence of x
    It applies: a < x < b.

    :param a: x-value for point a
    :type a: float
    :param b: x-value for point b
    :type b: float
    :param f_a: y-value for point a
    :type f_a: float
    :param f_b: y-value for point b
    :type f_b: float
    :param x: x-value for the searched answer f_x
    :type x: float
    :return: y-value for given x-value
    """
    slope = (f_b - f_a) / (b - a)
    f_x = slope * (x - a) + f_a
    return f_x


def my_polate_linear(a: float, b: float, f_a: float, f_b: float, x: float):
    """
    Interpolates or extrapolates linear for a<x<b or x<a and x>b.

    :param a: input x-value for point a
    :type a: float
    :param b: input x-value for point b
    :type b: float
    :param f_a: input y-value for point a
    :type f_a: float
    :param f_b: input y-value for point b
    :type f_b: float
    :param x: x-value for the searched answer f_x
    :type x: float
    :return: y-value for given x-value
    """
    if a == b == x and f_a == f_b:
        f_x = f_a
    else:
        slope = (f_b - f_a) / (b - a)
        f_x = slope * (x - a) + f_a
    return f_x


def interpolate_neighbours_linear(temperature: float, frequency: float, neighbours: dict):
    """
    Linear interpolation of frequency and temperature between neighbours.

    :param temperature: desired temperature value in °C
    :type temperature: float
    :param frequency: desired frequency value in Hz
    :type frequency: float
    :param neighbours: neighbours
    :type neighbours: dict
    :return: amplitude of the permittivity, angle of the permittivity in degree
    """
    # Interpolation of Amplitude
    # in temperature at f_low
    epsilon_r_at_T_f_low = my_polate_linear(a=neighbours["T_low_f_low"]["temperature"]["value"], b=neighbours["T_high_f_low"]["temperature"]["value"],
                                            f_a=neighbours["T_low_f_low"]["epsilon_r"], f_b=neighbours["T_high_f_low"]["epsilon_r"], x=temperature)
    # in temperature at f_high
    epsilon_r_at_T_f_high = my_polate_linear(a=neighbours["T_low_f_high"]["temperature"]["value"], b=neighbours["T_high_f_high"]["temperature"]["value"],
                                             f_a=neighbours["T_low_f_high"]["epsilon_r"], f_b=neighbours["T_high_f_high"]["epsilon_r"], x=temperature)
    # between f_low and f_high
    epsilon_r = my_polate_linear(a=neighbours["T_low_f_low"]["frequency"]["value"], b=neighbours["T_low_f_high"]["frequency"]["value"],
                                 f_a=epsilon_r_at_T_f_low, f_b=epsilon_r_at_T_f_high, x=frequency)

    # Interpolation of Phase
    # in temperature at f_low
    epsilon_phi_deg_at_T_f_low = my_polate_linear(a=neighbours["T_low_f_low"]["temperature"]["value"], b=neighbours["T_high_f_low"]["temperature"]["value"],
                                                  f_a=neighbours["T_low_f_low"]["epsilon_phi_deg"], f_b=neighbours["T_high_f_low"]["epsilon_phi_deg"],
                                                  x=temperature)
    # in temperature at f_high
    epsilon_phi_deg_at_T_f_high = my_polate_linear(a=neighbours["T_low_f_high"]["temperature"]["value"], b=neighbours["T_high_f_high"]["temperature"]["value"],
                                                   f_a=neighbours["T_low_f_high"]["epsilon_phi_deg"], f_b=neighbours["T_high_f_high"]["epsilon_phi_deg"],
                                                   x=temperature)
    # between f_low and f_high
    epsilon_phi_deg = my_polate_linear(a=neighbours["T_low_f_low"]["frequency"]["value"], b=neighbours["T_low_f_high"]["frequency"]["value"],
                                       f_a=epsilon_phi_deg_at_T_f_low, f_b=epsilon_phi_deg_at_T_f_high, x=frequency)

    return epsilon_r, epsilon_phi_deg
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# Add and remove data in Database


# General Permeability -----------------------------------------------------------------------------------------------------------------------------------------
def create_permeability_measurement_in_database(material_name: str, measurement_setup: str, company: str = "", date: str = "", test_setup_name: str = "",
                                                toroid_dimensions: str = "", measurement_method: str = "", equipment_names: str = "", comment: str = ""):
    """
    Create a new permeability section in the database for a material.

    :param material_name: name of the material
    :type material_name: str
    :param measurement_setup: name of the measurement setup
    :type measurement_setup: str
    :param company: name of the company
    :type company: str
    :param date: date of measurement
    :type date: str
    :param test_setup_name: information of the test setup
    :type test_setup_name: str
    :param toroid_dimensions: dimensions of the probe
    :type toroid_dimensions: str
    :param measurement_method: name of the measurement method
    :type measurement_method: str
    :param equipment_names: name of the measurement equipment
    :type equipment_names: str
    :param comment: comment regarding the measurement
    :type comment: str
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if material_name not in data:
        print(f"Material {material_name} does not exist in materialdatabase.")
    else:
        if "complex_permeability" not in data[material_name]["measurements"]:
            print("Create complex permeability measurement.")
            data[material_name]["measurements"]["complex_permeability"] = {}

    data[material_name]["measurements"]["complex_permeability"][measurement_setup] = {
        "data_type": "complex_permeability_data",
        "name": measurement_setup,
        "company": company,
        "date": date,
        "test_setup": {
            "name": test_setup_name,
            "Toroid": toroid_dimensions,
            "Measurement_Method": measurement_method,
            "Equipment": equipment_names,
            "comment": comment
        },
        "measurement_data": []
    }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def clear_permeability_measurement_data_in_database(material_name: str, measurement_setup: str):
    """
    Clear the permeability data in the database given a material and measurement setup.

    :param material_name: name of the material
    :type material_name: str
    :param measurement_setup: name of the measurement setup
    :type measurement_setup: str
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permeability_data_into_database(frequency: float, temperature: float, b_ref: np.ndarray | list, mu_r_abs: np.ndarray | list,
                                          mu_phi_deg: np.ndarray | list, material_name: str, measurement_setup: str, current_shape: str = "sine",
                                          H_DC_offset: float = 0, overwrite: bool = False):
    """
    Write permeability data into the material database.

    CAUTION: This method only adds the given measurement series to the permeability data without checking duplicates.

    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature: temperature value in °C
    :type temperature: float
    :param b_ref: magnetic flux density value
    :type b_ref: ndarray or list
    :param mu_r_abs: amplitude of the permeability
    :type mu_r_abs: ndarray or list
    :param mu_phi_deg: angle of the permeability
    :type mu_phi_deg: ndarray or list
    :param material_name: name of the material
    :type material_name: str
    :param measurement_setup: name of the measurement setup
    :type measurement_setup: str
    :param current_shape: shape of the current (e.g. "sine", "triangle", "trapezoid")
    :type current_shape: str
    :param H_DC_offset: offset in the magnetic field strength in A/m
    :type H_DC_offset: float
    :param overwrite: enable/disable overwritting of data
    :type overwrite: bool
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if "measurement_data" not in data[material_name]["measurements"]["complex_permeability"][measurement_setup]:
        data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    elif not isinstance(data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"], list):
        data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    if overwrite:
        # check if list is empty
        if len(data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"]) > 0:
            data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"].append(
        {
            "signal_shape": current_shape,
            "temperature": temperature,
            "frequency": frequency,
            "H_DC_offset": H_DC_offset,
            "flux_density": list(b_ref),
            "mu_r_abs": list(mu_r_abs),
            "mu_phi_deg": list(mu_phi_deg)
        }
    )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# General Steinmetz --------------------------------------------------------------------------------------------------------------------------------------------
def write_steinmetz_data_into_database(temperature: float, k: float, beta: float, alpha: float, material_name: str, measurement_setup: str):
    """
    Write steinmetz data into the material database.

    CAUTION: This method only adds the given measurement series to the steinmetz data without checking duplicates.

    :param temperature: temperature value in °C
    :type temperature: float
    :param k: k value of steinmetz parameters
    :type k: float
    :param beta: beta value of the steinmetz parameters
    :type beta: float
    :param alpha: alpha value of the steinmetz parameters
    :type alpha: float
    :param material_name: name of the material
    :type material_name: str
    :param measurement_setup: name of the measurement setup
    :type measurement_setup: str
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if measurement_setup not in data[material_name]["measurements"]["Steinmetz"]:
        data[material_name]["measurements"]["Steinmetz"][measurement_setup] = {
            "data_type": "steinmetz_data",
            "name": measurement_setup,
            "data": []
        }

    data[material_name]["measurements"]["Steinmetz"][measurement_setup]["data"].append(
        {
            "temperature": float(temperature),
            "k": k,
            "alpha": alpha,
            "beta": beta,
        }
    )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def create_empty_material(material_name: str, manufacturer: str, initial_permeability: float, resistivity: float,
                          max_flux_density: float, volumetric_mass_density: float):
    """
    Create an empty material slot in the database.

    :param material_name: name of the material
    :type material_name: str
    :param manufacturer: name of the manufacturer
    :type manufacturer: str
    :param initial_permeability: value of the initial permeability
    :type initial_permeability: float
    :param resistivity: value of the resistivity
    :type resistivity: float
    :param max_flux_density: saturation value of the magnetic flux density
    :type max_flux_density: float
    :param volumetric_mass_density: value of the volumetric mass density
    :type volumetric_mass_density: float
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if material_name in data:
        print(f"Material {material_name} already exists in materialdatabase.")
    else:
        data[material_name] = {
            "Manufacturer": manufacturer,
            "manufacturer_datasheet": {
                "initial_permeability": initial_permeability,
                "resistivity": resistivity,
                "max_flux_density": max_flux_density,
                "volumetric_mass_density": volumetric_mass_density
            },
            "measurements": {}
        }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# General Permittivity -----------------------------------------------------------------------------------------------------------------------------------------
def create_permittivity_measurement_in_database(material_name: str, measurement_setup: str, company: str = "", date: str = "", test_setup_name: str = "",
                                                probe_dimensions: str = "", measurement_method: str = "", equipment_names: str = "", comment: str = ""):
    """
    Create a new permittvity section in the database for a material.

    :param material_name: name of the material
    :type material_name: str
    :param measurement_setup: name of the measurement setup
    :type measurement_method: str
    :param company: name of the company
    :type company: str
    :param date: date of measurement
    :type date: str
    :param test_setup_name: information of the test setup
    :type test_setup_name: str
    :param probe_dimensions: dimensions of the probe
    :type probe_dimensions: str
    :param measurement_method: name of the measurement method
    :type measurement_method: str
    :param equipment_names: name of the measurement equipment
    :type equipment_names: str
    :param comment: comment regarding the measurement
    :type comment: str
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if material_name not in data:
        print(f"Material {material_name} does not exist in materialdatabase.")
    else:
        if "complex_permittivity" not in data[material_name]["measurements"]:
            print("Create complex permittivity measurement.")
            data[material_name]["measurements"]["complex_permittivity"] = {}
            data[material_name]["measurements"]["complex_permittivity"][measurement_setup] = {
                "data_type": "complex_permittivity_data",
                "name": measurement_setup,
                "company": company,
                "date": date,
                "test_setup": {
                    "name": test_setup_name,
                    "Probe": probe_dimensions,
                    "Measurement_Method": measurement_method,
                    "Equipment": equipment_names,
                    "comment": comment
                },
                "measurement_data": []
            }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def clear_permittivity_measurement_data_in_database(material_name: str, measurement_setup: str):
    """
    Clear the permittivity data in the database for a specific material.

    :param material_name: name of material
    :type material_name: str
    :param measurement_setup: name of measurement setup
    :type measurement_setup: str
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permittivity_data_into_database(temperature: float, frequencies: np.ndarray | list, epsilon_r: np.ndarray | list, epsilon_phi_deg: np.ndarray | list,
                                          material_name: str, measurement_setup: str):
    """
    Write permittivity data into the material database.

    :param temperature: measurement point of the temperature in °C
    :type temperature: float
    :param frequencies: measurement points of the frequency in Hz
    :type frequencies: ndarray or list
    :param epsilon_r: amplitude of the permittivity
    :type epsilon_r: ndarray or list
    :param epsilon_phi_deg: angle of the permittivity
    :type epsilon_phi_deg: ndarray or list
    :param material_name: name of material
    :type material_name: str
    :param measurement_setup: name of measurement setup
    :type measurement_setup: str
    """
    # load data

    # mean of data

    # write data in DB
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    # if type(data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"]) is not list:
    #     data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"] = []

    if temperature in set([element["temperature"] for element in
                           data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"]]):
        print(f"Temperature {temperature} C is already in database!")
    else:
        data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"].append(
            {
                "temperature": temperature,
                "frequencies": list(frequencies),
                "epsilon_r": list(epsilon_r),
                "epsilon_phi_deg": list(epsilon_phi_deg)
            }
        )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# LEA_LK Permeability ------------------------------------------------------------------------------------------------------------------------------------------
def get_permeability_data_from_lea_lk(location: str, frequency: float, temperature: float, material_name: str, no_interpolation_values: int = 20):
    """
    Get the permeability data from LEA_LK.

    :param location: location of the permeability data
    :type location: str
    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature: temperature value in °C
    :type temperature: float
    :param material_name: name of the material
    :type material_name: str
    :param no_interpolation_values: number of interpolation values
    :type no_interpolation_values: int
    :return: magnetic flux density, amplitude of the permeability, angle of the permeability
    """
    b_hys, p_hys = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="Core_Loss", quantity="p_hys", frequency=frequency,
                                                         material_name=material_name, temperature=temperature)
    b_phi, mu_phi_deg = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="mu_phi_Plot", quantity="mu_phi",
                                                              frequency=frequency, material_name=material_name, temperature=temperature)

    # Find the border of the common magnetic flux density values
    b_max_min = max(min(b_hys), min(b_phi), min(b_hys), min(b_phi))
    b_min_max = min(max(b_hys), max(b_phi), max(b_hys), max(b_phi))
    # mdb_print(f"{b_max_min, b_min_max = }")
    # Create the magnetic flux density vector that is used for later interpolation actions
    b_common = np.linspace(b_max_min, b_min_max, no_interpolation_values)

    f_p_hys_interpol = interp1d(b_hys, p_hys)
    f_b_phi_interpol = interp1d(b_phi, mu_phi_deg)
    f_p_hys_interpol_common = f_p_hys_interpol(b_common)
    f_b_phi_interpol_common = f_b_phi_interpol(b_common)

    # mdb_print(f"{b_common, f_p_hys_interpol_common, f_b_phi_interpol_common = }")

    return b_common, mu_r__from_p_hyst_and_mu_phi_deg(f_b_phi_interpol_common, frequency, b_common, f_p_hys_interpol_common), f_b_phi_interpol_common


def create_permeability_file_name_lea_lk(quantity: str = "p_hys", frequency: float = 100000, material_name: str = "N49", temperature: float = 30):
    """
    Create the file name for permeability data of LEA_LK.

    :param quantity: measured quantiy (e.g. p_hys)
    :type quantity: str
    :param material_name: name of the material
    :type material_name: str
    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature: temperature value in °C
    :type temperature: float
    :return: correct file name for LEA_LK
    """
    return "_".join([quantity, f"{int(frequency / 1000)}kHz", material_name, f"{temperature}C.txt"])


def get_permeability_property_from_lea_lk(path_to_parent_folder: str, quantity: str, frequency: float, material_name: str, temperature: float,
                                          sub_folder_name: str = "Core_Loss"):
    """
    Get the proberty of the permeability from LEA_LK.

    :param path_to_parent_folder: path to permeability data
    :type path_to_parent_folder: str
    :param quantity: name of the measured quantity
    :type quantity: str
    :param material_name: name of the material
    :type material_name: str
    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature: temperature value in °C
    :type temperature: float
    :param sub_folder_name: name of the sub folder
    :type sub_folder_name: str
    :return: amplitude of the permeability, angle of the permeability
    """
    filename = create_permeability_file_name_lea_lk(quantity, frequency, material_name, temperature)
    complete_path = os.path.join(path_to_parent_folder, sub_folder_name, filename)
    # mdb_print(complete_path)

    data = np.loadtxt(complete_path)
    # mdb_print(data)
    return data[:, 0], data[:, 1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# Permittivity -------------------------------------------------------------------------------------------------------------------------------------------------
def get_permittivity_data_from_lea_lk(location: str, temperature: float, frequency: float, material_name: str):
    """
    Get the permittivity data from LEA_LK.

    :param location: location of the permittivity data
    :type location: str
    :param temperature: temperature value
    :type temperature: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param material_name: name of the material
    :type material_name: str
    :return: amplitude of the permittivity, angle of the permittivity
    """
    e_amplitude, epsilon_r_tilde = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_r_Plot", quantity="eps_r_tilde",
                                                                         frequency=frequency, material_name=material_name, temperature=temperature)

    e_phi, epsilon_phi_deg = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_phi_Plot", quantity="eps_phi_tilde",
                                                                   frequency=frequency, material_name=material_name, temperature=temperature)

    return epsilon_r_tilde, epsilon_phi_deg


def create_permittivity_file_name_lea_lk(quantity: str = "p_hys", frequency: float = 100000, material_name: str = "N49", temperature: float = 30):
    """
    Create the file name for permittivity data of LEA_LK.

    :param quantity: measured quantiy (e.g. p_hys)
    :type quantity: str
    :param frequency: frequency value in Hz
    :type frequency: float
    :param material_name: name of the material
    :type material_name: str
    :param temperature: temperature value in °C
    :type temperature: float
    :return: correct file name for LEA_LK
    """
    return "_".join([quantity, material_name, f"{temperature}C", f"{int(frequency / 1000)}kHz.txt"])


def get_permittivity_property_from_lea_lk(path_to_parent_folder: str, quantity: str, frequency: float, material_name: str, temperature: float,
                                          sub_folder_name: str = "Core_Loss"):
    """
    Get the proberty of the permittivity from LEA_LK.

    :param path_to_parent_folder: path to permittivity data:
    :type path_to_parent_folder: str
    :param quantity: name of the measured quantity
    :type quantity: str
    :param frequency: frequency value in Hz
    :type frequency: float
    :param material_name: name of the material
    :type material_name: str
    :param temperature: temperature value in °C
    :type temperature: float
    :param sub_folder_name: name of the sub folder
    :type sub_folder_name: str
    :return: amplitude of the permittivity, angle of the permittivity
    """
    filename = create_permittivity_file_name_lea_lk(quantity, frequency, material_name, temperature)
    complete_path = os.path.join(path_to_parent_folder, sub_folder_name, filename)
    # mdb_print(complete_path)

    data = np.loadtxt(complete_path)
    # mdb_print(data)
    return data[:, 0], data[:, 1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# LEA_MTB ------------------------------------------------------------------------------------------------------------------------------------------------------
def get_permeability_property_from_lea_mtb(path_to_parent_folder: str):
    """
    Get the proberty of the permeability from the material test bench.

    :param path_to_parent_folder: path to permeability data
    :type path_to_parent_folder: str
    :return: magnetic flux density, amplitude of the permeability, angle of the permeability
    """
    # hardcode: select the first file available in the directory
    # TODO: find a better solution
    new_list = [lis for lis in os.listdir(path_to_parent_folder) if '.csv' in lis]
    data = np.genfromtxt(os.path.join(path_to_parent_folder, new_list[0]), delimiter=',', skip_header=True)
    return data[:, 4], abs(data[:, 2]), abs(data[:, 3])


def get_permeability_data_from_lea_mtb(location: str):  # TODO: IS THIS FUNCTION NECESSARY, ONLY CALLS THE UPPER FUNCTION WITHOUT ADDITIONAL FUNCTIONALTY
    """
    Get the permeability data from the material test bench.

    :param location: location of the permability data
    :type location: str
    :return: magnetic flux density, amplitude of the permeability, angle of the permeability
    """
    b_hys, mu_r_abs, mu_phi_deg = get_permeability_property_from_lea_mtb(path_to_parent_folder=location)

    return b_hys, mu_r_abs, mu_phi_deg


def get_all_frequencies_for_material(material_path: str):
    """
    Get all the frequency values for a given material.

    :param material_path: path to the material
    :type material_path: str
    :return: all frequency values in Hz of the given material
    """
    frequencies_str = os.listdir(material_path)
    print(frequencies_str)
    frequencies = []
    for f_str in frequencies_str:
        if "kHz" in f_str:
            frequencies.append(int(f_str[0:-3]) * 1000)
    print(frequencies)
    return frequencies


def get_all_temperatures_for_directory(toroid_path: str):
    """
    Get all the temperature values for a given toroid probe.

    :param toroid_path: path of the toroid probe
    :type toroid_path: str
    :return: all temperature values in °C of the specific toroid probe
    """
    temperatures_str = os.listdir(toroid_path)
    temperatures = []
    for f_str in temperatures_str:
        try:
            temperatures.append(int(f_str))
        except:
            pass
    return temperatures


def sigma_from_permittivity(amplitude_relative_equivalent_permittivity: np.ndarray | float, phi_deg_relative_equivalent_permittivity: np.ndarray | float,
                            frequency: np.ndarray | float):
    """
    Calculate the conductivity based on the data of the permittivity.

    :param amplitude_relative_equivalent_permittivity: amplitude of the permittivity
    :type amplitude_relative_equivalent_permittivity: ndarray or float
    :param phi_deg_relative_equivalent_permittivity: angle of the permittivity
    :type phi_deg_relative_equivalent_permittivity: ndarray or float
    :param frequency: frequency value in Hz
    :type frequency: nd array or float
    :return: conductivity
    """
    return 2 * np.pi * frequency * amplitude_relative_equivalent_permittivity * epsilon_0 * j * \
        (np.cos(np.deg2rad(phi_deg_relative_equivalent_permittivity)) + j * np.sin(np.deg2rad(phi_deg_relative_equivalent_permittivity)))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# mathematical functions----------------------------------------------------------------------------------------------------------------------------
def remove_mean_of_signal(signal: np.ndarray = None):
    """
    Remove the mean value of a signal.

    :param signal: signal with mean value
    :type signal: ndarray
    :return: signal without mean value
    """
    return signal - (max(signal) - (max(signal) - min(signal)) / 2)


def integrate(x_data: np.ndarray | list, y_data: np.ndarray | list):
    """
    Integrate the function y_data = f(x_data).

    :param x_data: x-axis
    :type x_data: ndarray or list
    :param y_data: y-axis
    :type y_data: ndarray or list
    :return: defined integral of y_data
    """
    data = [np.trapz(y_data[0:index], x_data[0:index]) for index, value in enumerate(x_data)]
    return np.array(data)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# calculation functions magnetic field--------------------------------------------------------------------------------------------------------------------------
def mu_r__from_p_hyst_and_mu_phi_deg(mu_phi_deg: np.ndarray | float, frequency: np.ndarray | float, b_peak: np.ndarray | float, p_hyst: np.ndarray | float):
    """
    Calculate the amplitude of the permeability given the peak value of the magnetic flux density, the hysteresis loss and the phase angle of the permeability.

    :param mu_phi_deg: phase angle of the permeability in degree
    :type mu_phi_deg: ndarray or float
    :param frequency: frequency in Hz
    :type frequency: ndarray or float
    :param b_peak: peak flux density in T
    :type b_peak: ndarray or float
    :param p_hyst: hysteresis losses in W/m^3
    :type p_hyst: ndarray or float
    :return: amplitude of the permeability
    """
    b_peak = np.array(b_peak)
    return b_peak ** 2 * np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) / p_hyst / mu_0


def p_hyst__from_mu_r_and_mu_phi_deg(frequency: np.ndarray | float, b_peak: np.ndarray | float, mu_r: np.ndarray | float, mu_phi_deg: np.ndarray | float):
    """
    Calculate the hysteresis losses given the peak value of the magnetic flux density, the amplitude and phase angle of the permeability.

    :param frequency: frequency in Hz
    :type frequency: ndarray or float
    :param b_peak: peak flux density in T
    :type b_peak: ndarray or float
    :param mu_r: amplitude of the permeability in unitless
    :type mu_r: ndarray or float
    :param mu_phi_deg: phase angle of the permeability in degree
    :type mu_phi_deg: ndarray or float
    :return: hysteresis losses in W/m^3
    """
    b_peak = np.array(b_peak)
    return np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) * mu_0 * mu_r * (b_peak / mu_0 / mu_r) ** 2


def mu_phi_deg__from_mu_r_and_p_hyst(frequency: np.ndarray | float, b_peak: np.ndarray | float, mu_r: np.ndarray | float, p_hyst: np.ndarray | float):
    """
    Calculate the phase angle of the permeability given the peak value of the magnetic flux density, the hysteresis loss and the amplitude of permeability.

    :param frequency: frequency in Hz
    :type frequency: ndarray or float
    :param b_peak: peak flux density in T
    :type b_peak: ndarray or float
    :param mu_r: amplitude of the permeability in unitless
    :type mu_r: ndarray or float
    :param p_hyst: hysteresis losses in W/m^3
    :type p_hyst: ndarray or float
    :return: phase angle of the permeability in degree
    """
    b_peak = np.array(b_peak)
    return np.rad2deg(np.arcsin(p_hyst * mu_r * mu_0 / (np.pi * frequency * b_peak ** 2)))


def get_bh_integral_shoelace(b: np.ndarray, h: np.ndarray, f: float):
    """
    Calculate the hysteresis loss density.

    :param b: magnetic flux density in T
    :type b: ndarray
    :param h: magnetic field strength in A/m
    :type h: ndarray
    :param f: frequency in Hz
    :type f: float
    :return: hysteresis loss density in W/m^3
    """
    return f * 0.5 * np.abs(np.sum(b * (np.roll(h, 1, axis=0) - np.roll(h, -1, axis=0)), axis=0))  # shoelace formula


def get_bh_integral_trapezoid(b: np.ndarray, h: np.ndarray, f: float):
    """
    Calculate the hysteresis loss density.

    :param b: magnetic flux density in T
    :type b: ndarray
    :param h: magnetic field strength in A/m
    :type h: ndarray
    :param f: frequency in Hz
    :type f: float
    :return: hysteresis loss density in W/m^3
    """
    return f * np.trapezoid(h * np.gradient(b))


def calc_magnetic_flux_density_based_on_voltage_array_and_frequency(voltage: np.ndarray | list, frequency: float = 1.0, secondary_winding: int = 1,
                                                                    cross_section: float = 1.0):
    """
    Calculate the magnetic flux density based on the voltage and the frequency.

    ASSUMPTION: EXACTLY ONE PERIOD!
    Based on the length of the voltage array and the frequency the time-array is constructed for the integration.

    :param voltage: array-like of the voltage in V
    :type voltage: ndarray or list
    :param frequency: frequency value in Hz
    :type frequency: float
    :param secondary_winding: number of secondary windings
    :type secondary_winding: int
    :param cross_section: value of the cross-section of the core in m^2
    :type cross_section: float
    :return: array with magnetic flux density curve in T
    """
    voltage = np.array(voltage)
    time = np.linspace(0, 1/frequency, voltage.shape[0])
    return integrate(time, voltage) / secondary_winding / cross_section


def calc_magnetic_flux_density_based_on_voltage_array_and_time_array(voltage: np.ndarray | list, time: np.ndarray | list, secondary_winding: int = 1,
                                                                     cross_section: float = 1):
    """
    Calculate the magnetic flux density based on the voltage and the time.

    ASSUMPTION: EXACTLY ONE PERIOD!

    :param voltage: array-like of the voltage in V
    :type voltage: ndarray or list
    :param time: array-like of the time in s
    :type time: ndarray or list
    :param secondary_winding: number of secondary windings
    :type secondary_winding: int
    :param cross_section: value of the cross-section of the core in m^2
    :type cross_section: float
    :return: array with magnetic flux density curve in T
    """
    voltage, time = np.array(voltage), np.array(time)
    return integrate(time, voltage) / secondary_winding / cross_section


def calc_magnetic_field_strength_based_on_current_array(current: np.ndarray | list, primary_winding: int = 1, l_mag: float = 1.0):
    """
    Calculate the magnetic field strength based on the current.

    :param current: array-like of the current in A
    :type current: ndarray or list
    :param primary_winding: number of primary windings
    :type primary_winding: int
    :param l_mag: mean magnetic path length in m
    :type l_mag: float
    :return: array with magnetic field strength curve in A/m
    """
    current = np.array(current)
    return current * primary_winding / l_mag


def calc_mu_r_from_b_and_h_array(b: np.ndarray | list, h: np.ndarray | list):
    """
    Calculate the amplitude of the relative permeability based on a magnetic flux density and a magnetic field strength array.

    :param b: magnetic flux density array
    :type b: ndarray or list
    :param h: magnetic field strength array
    :type h: ndarray or list
    :return: amplitude of the relative permability
    """
    b, h = np.array(b), np.array(h)
    mu_r = ((max(b) - min(b))/2) / ((max(h) - min(h))/2) / mu_0
    return mu_r
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# calculation functions electric field -------------------------------------------------------------------------------------------------------------------------
def calc_electric_flux_density_based_on_current_array_and_frequency(current: np.ndarray | list, frequency: float = 1.0, cross_section: float = 1.0)\
        -> np.ndarray:
    """
    Calculate the electric flux density based on the current and the frequency.

    :param current: array-like of the current in A
    :type current: np.ndarray or list
    :param frequency: frequency value in Hz
    :type frequency: float
    :param cross_section: cross-section of probe in m^2
    :type cross_section: float
    :return: electric flux density values
    :rtype: np.ndarray
    """
    current = np.array(current)
    time = np.linspace(0, 1 / frequency, current.shape[0])
    electric_flux_density = integrate(time, current) / cross_section
    return electric_flux_density


def calc_electric_flux_density_based_on_current_array_and_time_array(current: np.ndarray | list, time: np.ndarray | list, cross_section: float = 1.0)\
        -> np.ndarray:
    """
    Calculate the electric flux density based on the current data of a lecroy oscilloscope.

    :param current: array-like of the current in A
    :type current: np.ndarray or list
    :param time: array of time values of measurement in s
    :type time: np.ndarray or list
    :param cross_section: cross-section of probe in m^2
    :type cross_section: float
    :return: electric flux density values
    :rtype: np.ndarray
    """
    electric_flux_density = integrate(time, current) / cross_section
    return electric_flux_density


def calc_electric_field_strength_from_lecroy_voltage_data(voltage: np.ndarray | list, height: float = 1.0) -> np.ndarray:
    """
    Calculate the electric field strength based on the voltage data of a lecroy oscilloscope.

    :param voltage: array-like of the voltage in V
    :type voltage: np.ndarray or list
    :param height: height of probe in m
    :type height: float
    :return: electric field strength values
    :rtype: np.ndarray
    """
    electric_field_strength = np.array(voltage) / height
    return electric_field_strength


def eps_phi_deg__from_eps_r_and_p_eddy(frequency: np.ndarray | float, e_peak: np.ndarray | float, eps_r: np.ndarray | float, p_eddy: np.ndarray | float)\
        -> np.ndarray:
    """
    Calculate the angle of the permittivity.

    :param frequency: frequency
    :type frequency: np.ndarray or float
    :param e_peak: peak value of the electric field strength
    :type e_peak: np.ndarray or float
    :param eps_r: peak value of the amplitude of the permittivity
    :type eps_r: np.ndarray or float
    :param p_eddy: eddy current loss density
    :type p_eddy: np.ndarray or float
    :return: angle of permittivity in degree
    :rtype: np.ndarray
    """
    return np.rad2deg(np.arcsin(p_eddy / (np.pi * frequency * eps_r * epsilon_0 * np.array(e_peak) ** 2)))


def calc_eps_r_from_d_and_e_array(d: np.ndarray | list, e: np.ndarray | list) -> np.ndarray:
    """
    Calculate the amplitude of the relative permeability based on a magnetic flux density and a magnetic field strength array.

    :param d: electric flux density array
    :type d: np.ndarray or list
    :param e: electric field strength array
    :type e: np.ndarray or list
    :return: amplitude of the relative permittivity
    :rtype: np.ndarray
    """
    d, e = np.array(d), np.array(e)
    eps_r = ((max(d) - min(d))/2) / ((max(e) - min(e))/2) / epsilon_0
    return eps_r
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# unused or externally used ------------------------------------------------------------------------------------------------------------------------------------
def find_nearest_frequencies(permeability: np.ndarray | list, frequency: float):
    """
    Find the nearest frequency value for permeability data.

    :param permeability: permeability data
    :type permeability: ndarray or list
    :param frequency: desired frequency value in Hz
    :type frequency: float
    :return: two frequency values in Hz with the desired value in between
    """
    freq_list = []
    # mdb_print(f"{freq_list = }")
    for i in range(len(permeability)):
        freq_list.append(permeability[i]["frequency"])
    # mdb_print(f"{freq_list = }")

    freq_list = list(remove(freq_list, len(freq_list)))
    # mdb_print(f"{freq_list = }")

    result = find_nearest(freq_list, frequency)

    return result[0], result[1]


def find_nearest_temperatures(permeability: np.ndarray | list, f_l: float, f_h: float, temperature: float):
    """
    Find the nearest temperature value between two frequency points.

    :param permeability: permeability data
    :type permeability: ndarray or list
    :param f_l: lower frequency value in Hz
    :type f_l: float
    :param f_h: higher frequency value in Hz
    :type f_h: float
    :param temperature: desired temperature value in °C
    :type temperature: float
    :return: two temperature values in °C with the desired value in between
    """
    # ------find nearby temperature------
    temp_list_l = []
    temp_list_h = []

    for i in range(len(permeability)):
        if permeability[i]["frequency"] == f_l:
            temp_list_l.append(permeability[i]["temperature"])
    for i in range(len(permeability)):
        if permeability[i]["frequency"] == f_h:
            temp_list_h.append(permeability[i]["temperature"])

    return find_nearest(temp_list_l, temperature), find_nearest(temp_list_h, temperature)


def getdata_measurements(permeability: np.ndarray | list, variable: float, frequency: float, temperature_1: float, temperature_2: float,
                         b_t: np.ndarray | list):
    """
    Linear interpolation of the permeability data between two temperatures at a constant frequency.

    :param permeability: permeability data
    :type permeability: ndarray or list
    :param variable: desired temperature variable in °C
    :type variable: float
    :param frequency: frequency value in Hz
    :type frequency: float
    :param temperature_1: temperature value under the desired value in °C
    :type temperature_1: float
    :param temperature_2: temperature value above the desired value in °C
    :type temperature_2: float
    :param b_t: magnetic flux density
    :type b_t: ndarray or list
    :return: amplitude of the permeability, angle of the permeability
    """
    for k in range(len(permeability)):
        if permeability[k]["frequency"] == frequency and permeability[k]["temperature"] == temperature_1:
            t_mu_phi_1 = interp1d(permeability[k]["flux_density"], permeability[k]["mu_phi_deg"])
            t_mu_r_1 = interp1d(permeability[k]["mu_r_abs"], permeability[k]["mu_r_abs"])

        if permeability[k]["frequency"] == frequency and permeability[k]["temperature"] == temperature_2:
            t_mu_phi_2 = interp1d(permeability[k]["flux_density"], permeability[k]["mu_phi_deg"])
            t_mu_r_2 = interp1d(permeability[k]["mu_r_abs"], permeability[k]["mu_r_abs"])
    # --------linear interpolation at constant freq-------------
    mu_phi = []
    mu_r = []

    for y in range(len(b_t)):
        mu_r.append(t_mu_r_1(b_t[y]) + (t_mu_r_2(b_t[y]) - t_mu_r_1(b_t[y])) / (temperature_2 - temperature_1) * (variable - temperature_1))
        mu_phi.append(t_mu_phi_1(b_t[y]) + (t_mu_phi_2(b_t[y]) - t_mu_phi_1(b_t[y])) / (temperature_2 - temperature_1) * (variable - temperature_1))
    return mu_r, mu_phi


def export_data(parent_directory: str = "", file_format: str = None, b_ref_vec: np.ndarray | list = None, mu_r_real_vec: np.ndarray | list = None,
                mu_r_imag_vec: np.ndarray | list = None, silent: bool = False):
    """
    Export data from the material database in a certain file format.

    :param parent_directory: path to parent directory
    :type parent_directory: str
    :param file_format: export format, e.g. 'pro' to export a .pro-file
    :type file_format: str
    :param b_ref_vec: reference vector for mu_r_real and mu_r_imag
    :type b_ref_vec: ndarray or list
    :param mu_r_real_vec: real part of mu_r_abs as a vector
    :type mu_r_real_vec: ndarray or list
    :param mu_r_imag_vec: imaginary part of mu_r_abs as a vector
    :type mu_r_imag_vec: ndarray or list
    :param silent: enables/disables print
    :type silent: bool
    """

    # fix numpy array inside normal python list problem
    # converts everything from scratch to a list, unified file format.
    b_ref_vec = np.array(b_ref_vec).tolist()
    mu_r_real_vec = np.array(mu_r_real_vec).tolist()
    mu_r_imag_vec = np.array(mu_r_imag_vec).tolist()

    if file_format == "pro":
        with open(os.path.join(parent_directory, "core_materials_temp.pro"), "w") as file:
            file.write('Include "Parameter.pro";\n')
            file.write(
                f"Function{{\n  b = {str(b_ref_vec).replace('[', '{').replace(']', '}')} ;\n  "
                f"mu_real = {str(mu_r_real_vec).replace('[', '{').replace(']', '}')} ;"
                f"\n  mu_imag = {str(mu_r_imag_vec).replace('[', '{').replace(']', '}')} ;\n  "
                f"mu_imag_couples = ListAlt[b(), mu_imag()] ;\n  "
                f"mu_real_couples = ListAlt[b(), mu_real()] ;\n  "
                f"f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{{List[mu_imag_couples]}};\n  "
                f"f_mu_real_d[] = InterpolationLinear[Norm[$1]]{{List[mu_real_couples]}};\n  "
                f"f_mu_imag[] = f_mu_imag_d[$1];\n  "
                f"f_mu_real[] = f_mu_real_d[$1];\n }}  ")

    else:
        raise Exception("No valid file format is given!")

    if not silent:
        print(f"Data is exported to {parent_directory} in a {file_format}-file.")


def plot_data(material_name: str = None, properties: str = None, b_ref: np.ndarray | list = None,
              mu_r_real: np.ndarray | list = None, mu_r_imag: np.ndarray | list = None):
    """
    Plot certain material properties of materials.

    TODO: parameter is new and will probably cause problems when plotting data, but previous implementation was very static...
    :param material_name: name of the material
    :type material_name: str
    :param properties: name of the material properties
    :type properties: str
    :param b_ref: magnetic flux density value:
    :type b_ref: ndarray or list
    :param mu_r_real: real part of the permeability
    :type mu_r_real: ndarray or list
    :param mu_r_imag: imaginary part of the permeability
    :type mu_r_imag: ndarray or list
    """
    if properties == "mu_r_real":
        plt.plot(b_ref, mu_r_real)
        plt.ylabel(properties)
        plt.xlabel('B in T')
        plt.title("Real part of permeability")
        plt.show()
    elif properties == "mu_r_imag":
        plt.plot(b_ref, mu_r_imag)
        plt.ylabel(properties)
        plt.xlabel('B in T')
        plt.title("Imaginary part of permeability")
        plt.show()

    print(f"Material properties {properties} of {material_name} are plotted.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
