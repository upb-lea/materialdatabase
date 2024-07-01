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


def remove(arr, n):
    """
    Remove duplicates from array.

    :param arr: array with duplicates
    :param n: has no effect of the functionality
    :return: array without duplicates
    """
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


def crop_data_fixed(x, pre_cropped_values: int = 0, post_cropped_values: int = 0):
    """
    Crop an array based on the given indices.

    IMPORTANT! THE SECOND INDEX IS COUNTED BACKWARDS(NEGATIVE)!

    :param x: array to get cropped
    :param pre_cropped_values: start value
    :param post_cropped_values: end value, but counted backwards
    :return: cropped data
    """
    if post_cropped_values == 0:
        post_cropped_values = -len(x)
    return x[pre_cropped_values:-post_cropped_values]


def crop_3_with_1(x, y, z, xa, xb):
    """
    Crop three arrays based on one array.

    :param x: array crop is based on
    :param y: first array to get cropped
    :param z: second array to get cropped
    :param xa: start value of crop
    :param xb: end value of crop
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
    :return: None
    :rtype: None
    """
    with open('material_data_base.json', 'w') as outfile:
        json.dump(data_to_be_stored, outfile, indent=4)
    print(f"Material properties of {material_name} are stored in the material database.")


def load_material_from_db(material_name: str) -> None:
    """
    Load data from material database.

    :param material_name: name of material
    :type material_name: str
    :return: None
    :rtype: None
    """
    with open(relative_path_to_db, 'r') as database:
        print("Read data from the data base.")
        data_dict = json.load(database)
    return data_dict[material_name]


def find_nearest(array, value):
    """
    Find the nearest value in an array.

    :param array: array to search
    :param value: desired value
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


def rect(radius_or_amplitude: float, theta_deg: float):
    """
    Convert polar coordinates [radius, angle] into cartesian coordinates [abscissa_x,ordinate_y].

    :param radius_or_amplitude: radius or amplitude
    :type radius_or_amplitude: float
    :param theta_deg: angle in degree
    :type theta_deg: float
    :return: tuple; (float, float); (abscissa_x,ordinate_y)
    """
    abscissa_x = radius_or_amplitude * np.cos(np.radians(theta_deg))
    ordinate_y = radius_or_amplitude * np.sin(np.radians(theta_deg))
    return abscissa_x, ordinate_y


def sort_data(a, b, c):
    """
    Sort three arrays according to array a.

    :param a: array that is the base of the sorting
    :param b: array that gets sorted based on a
    :param c: array that gets sorted based on a
    :return: the three arrays sorted according to a
    """
    sorted_list_of_lists = [list(x) for x in list(zip(*sorted(zip(a, b, c), key=lambda a: a)))]
    return np.array(sorted_list_of_lists[0]), np.array(sorted_list_of_lists[1]), np.array(sorted_list_of_lists[2])


def interpolate_a_b_c(a, b, c, no_interpolation_values=20):
    """
    Interpolation between three arrays based on the first array.

    :param a: array that is the base of the interpolation
    :param b: array that gets interpolated based on the values of array a
    :param c: array that gets interpolated based on the values of array a
    :param no_interpolation_values: number of interpolation values
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
def updates_x_ticks_for_graph(x_data: list, y_data: list, x_new: list):
    """
    Update the x-values of the given (x_data,y_data)-dataset and returns y_new based on x_new.

    :param x_data: x-data given
    :param y_data: y-data given
    :param x_new: new x-values
    :return: y_new-data corresponding to the x_new-data
    """
    f_linear = interp1d(x_data, y_data, fill_value="extrapolate")
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
    :param temperature: temperature in degree
    :type temperature: float
    :param frequency: frequency in Hz
    :type frequency: float
    :return: None
    :rtype: None
    """
    if datasource != MaterialDataSource.Measurement and datasource != MaterialDataSource.ManufacturerDatasheet:
        raise Exception("'datasource' must be 'manufacturer_datasheet' or 'measurements'.")

    if material_name is None or temperature is None or frequency is None:
        raise Exception(f"Failure in selecting data from materialdatabase. {material_name=}, {temperature=}, {frequency=}.")


def getdata_datasheet(permeability, variable, frequency, temperature_1, temperature_2):
    """
    Interpolation of permeability data between two temperatures at a constant frequency.

    Linear Interpolation between temperature_1 and temperature_2 to get a value for the temperature "variable".

    :param permeability: permeability data
    :param variable: desired temperature value in degree
    :param frequency: frequency value in Hz
    :param temperature_1: first temperature value in degree
    :param temperature_2: second temperature value
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


def create_permeability_neighbourhood_datasheet(temperature, frequency, list_of_permeability_dicts):
    """
    Create a neighbourhood for permeability data of a datasheet.

    :param temperature: temperature value in degree
    :param frequency: frequency value in Hz
    :param list_of_permeability_dicts: list of permeability data dicts
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_real": None,
                "mu_r_imag": None
            },
        "T_low_f_high":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_real": None,
                "mu_r_imag": None
            },
        "T_high_f_low":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_real": None,
                "mu_r_imag": None
            },
        "T_high_f_high":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_real": None,
                "mu_r_imag": None
            }
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


def create_permeability_neighbourhood_measurement(temperature, frequency, list_of_permeability_dicts):
    """
    Create a neighbourhood for permeability data of a measurement.

    :param temperature: temperature value in degree
    :param frequency: frequency value in Hz
    :param list_of_permeability_dicts: list of permeability dicts
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_abs": None,
                "mu_phi_deg": None
            },
        "T_low_f_high":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_abs": None,
                "mu_phi_deg": None
            },
        "T_high_f_low":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_abs": None,
                "mu_phi_deg": None
            },
        "T_high_f_high":
            {
                "index": None,
                "temperature": None,
                "frequency": None,
                "flux_density": None,
                "mu_r_abs": None,
                "mu_phi_deg": None
            }
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


def find_nearest_neighbour_values_permeability(permeability_data, temperature, frequency):
    """
    Find the nearest temperature and frequency values for a given neighbourhood of permeability data.

    :param permeability_data: permeability data
    :param temperature: temperature value in degree
    :param frequency: frequency value in Hz
    :return: lower temperature value in degree, higher temperature value in degree, lower frequency value in Hz, higher frequency value in Hz
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


def interpolate_b_dependent_quantity_in_temperature_and_frequency(temperature, frequency, temperature_low, temperature_high, frequency_low, frequency_high,
                                                                  b_t_low_f_low, f_b_T_low_f_low, b_T_high_f_low, f_b_T_high_f_low,
                                                                  b_T_low_f_high, f_b_T_low_f_high, b_T_high_f_high, f_b_T_high_f_high,
                                                                  no_interpolation_values: int = 8, y_label: str = None, plot: bool = False):
    """
    Interpolate a magnet flux density dependent quantity in temperature and frequency.

    :param temperature: desired temperature in degree
    :param frequency: desired frequency  in Hz
    :param temperature_low: lower temperature value in degree
    :param temperature_high: higher temperature value in degree
    :param frequency_low: lower frequency value in Hz
    :param frequency_high: higher frequency value in Hz
    :param b_t_low_f_low: magnetic flux density at the lower temperature in degree and the lower frequency value in Hz
    :param f_b_T_low_f_low: function dependent of b at the lower temperature in degree and the lower frequency value in Hz
    :param b_T_high_f_low: magnetic flux density at the higher temperature in degree and the lower frequency value in Hz
    :param f_b_T_high_f_low: function dependent of b at the higher temperature in degree and the lower frequency value in Hz
    :param b_T_low_f_high: magnetic flux density at the lower temperature in degree and the higher frequency value in Hz
    :param f_b_T_low_f_high: function dependent of b at the lower temperature in degree and the higher frequency value in Hz
    :param b_T_high_f_high: magnetic flux density at the higher temperature in degree and the higher frequency value in Hz
    :param f_b_T_high_f_high: function dependent of b at the higher temperature in degree and the higher frequency value in Hz
    :param no_interpolation_values: number of interpolation values
    :param y_label: label of y-axes
    :param plot: enable/disable plotting of data
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


def mu_r__from_p_hyst_and_mu_phi_deg(mu_phi_deg, frequency, b_peak, p_hyst):
    """
    Calculate the amplitude of the permeability given the peak value of the magnetic flux density, the hysteresis loss and the phase angle of the permeability.

    :param mu_phi_deg: phase angle of the permeability in degree
    :param frequency: frequency in Hz
    :param b_peak: peak flux density in T
    :param p_hyst: hysteresis losses in W/m^3
    :return: amplitude of the permeability
    """
    b_peak = np.array(b_peak)
    return b_peak ** 2 * np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) / p_hyst / mu_0


def p_hyst__from_mu_r_and_mu_phi_deg(frequency, b_peak, mu_r, mu_phi_deg):
    """
    Calculate the hysteresis losses given the peak value of the magnetic flux density, the amplitude and phase angle of the permeability.

    :param frequency: frequency in Hz
    :param b_peak: peak flux density in T
    :param mu_r: amplitude of the permeability in unitless
    :param mu_phi_deg: phase angle of the permeability in degree
    :return: hysteresis losses in W/m^3
    """
    b_peak = np.array(b_peak)
    return np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) * mu_0 * mu_r * (b_peak / mu_0 / mu_r) ** 2


def mu_phi_deg__from_mu_r_and_p_hyst(frequency, b_peak, mu_r, p_hyst):
    """
    Calculate the phase angle of the permeability given the peak value of the magnetic flux density, the hysteresis loss and the amplitude of permeability.

    :param frequency: frequency in Hz
    :param b_peak: peak flux density in T
    :param mu_r: amplitude of the permeability in unitless
    :param p_hyst: hysteresis losses in W/m^3
    :return: phase angle of the permeability in degree
    """
    b_peak = np.array(b_peak)
    return np.rad2deg(np.arcsin(p_hyst * mu_r * mu_0 / (np.pi * frequency * b_peak ** 2)))


def process_permeability_data(b_ref_raw, mu_r_raw, mu_phi_deg_raw, b_min: float = 0.05, b_max: float = 0.3, smooth_data: bool = False, crop_data: bool = False,
                              plot_data: bool = False, ax=None, f=None, T=None):
    """
    Post-Processing of raw data of the permeability.

    Function can smooth, crop and plot the permeability data.

    :param T: temperature value in degree
    :param f: frequency value in Hz
    :param ax: matplotlib axes for plotting
    :param b_max: max value of the magnetic flux density for cropping
    :param b_min: min value of the magnetic flux density for cropping
    :param b_ref_raw: raw data of the magnetic flux density
    :param mu_r_raw: raw data of the amplitude of the permeability
    :param mu_phi_deg_raw: raw data of the angle of the permeability
    :param smooth_data: enable/disable smoothing of data (savgol-filter)
    :param crop_data: enable/disable cropping of data
    :param plot_data: enable/disable plotting of data
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
        # fig, ax = plt.subplots(3)

        # Amplitude Plot
        # ax[0].plot(b_ref_raw, mu_r_raw, label=f)
        ax[0].plot(1000*b_ref, mu_r, "-x", label=f"{f=}, {T=}")
        ax[0].grid(True)
        ax[0].set_ylabel("rel. permeability")

        # Phase Plot
        # ax[1].plot(b_ref_raw, mu_phi_deg_raw)
        if smooth_data:
            ax[1].plot(1000*b_ref, mu_phi_deg, "-x", label=f"{f=}, {T=}")
        ax[1].grid(True)
        ax[1].legend()

        ax[1].set_ylabel("loss angle in deg")

        # Loss Plot
        loss_density = p_hyst__from_mu_r_and_mu_phi_deg(frequency=f, b_peak=b_ref, mu_r=mu_r, mu_phi_deg=mu_phi_deg)
        ax[2].loglog(1000*b_ref, loss_density/1000, label=f"{f=}, {T=}")
        ax[2].grid(which="both", ls="-")
        ax[2].set_xlabel("magnetic flux density in mT")
        ax[2].set_ylabel("loss density in kW/m^3")
        # ax[2].legend()
        # plt.show()

    return b_ref, mu_r, mu_phi_deg
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load Permittivity --------------------------------------------------------------------------------------------------------------------------------------------
def find_nearest_neighbours(value, list_to_search_in):
    """
    Return the two values with the wanted value in between and additional the indices of the corresponding values.

    Only works for sorted lists (small to big).

    Case 0: if len(list_to_search_in) == 1: return duplicated
    Case 1: if value == any(list_to_search_in): return duplicated
    Case 2: if value inbetween: return neighbours
    Case 3a: value smaller than data: return smallest two
    Case 3b: if value is bigger than data: return biggest two

    :param value: desired value
    :param list_to_search_in: array to search for value
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


def create_permittivity_neighbourhood(temperature, frequency, list_of_permittivity_dicts):
    """
    Create neighbourhood for permittivity data.
    
    :param temperature: temperature value in degree
    :param frequency: frequency value in Hz
    :param list_of_permittivity_dicts: list of permittivity data dicts
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "frequency": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_low_f_high":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "frequency": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_high_f_low":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "frequency": {
                    "value": None,
                    "index": None
                },
                "epsilon_r": None,
                "epsilon_phi_deg": None
            },
        "T_high_f_high":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "frequency": {
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


def create_steinmetz_neighbourhood(temperature, list_of_steinmetz_dicts):
    """
    Create neighbourhood for steinmetz data.

    :param temperature: temperature value in degree
    :param list_of_steinmetz_dicts: list of steinmetz data dicts
    :return: neighbourhood
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "k": None,
                "alpha": None,
                "beta": None
            },
        "T_high":
            {
                "temperature": {
                    "value": None,
                    "index": None
                },
                "k": None,
                "alpha": None,
                "beta": None
            }
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


def my_interpolate_linear(a, b, f_a, f_b, x):
    """
    Interpolates linear between to points 'a' and 'b'.

    The return value is f_x in dependence of x
    It applies: a < x < b.

    :param a: input x-value for point a
    :param b: input x-value for point b
    :param f_a: input y-value for point a
    :param f_b: input y-value for point b
    :param x: x-value for the searched answer f_x
    :return: y-value for given x-value
    """
    slope = (f_b - f_a) / (b - a)
    f_x = slope * (x - a) + f_a
    return f_x


def my_polate_linear(a, b, f_a, f_b, x):
    """
    Interpolates or extrapolates linear for a<x<b or x<a and x>b.

    :param a: input x-value for point a
    :param b: input x-value for point b
    :param f_a: input y-value for point a
    :param f_b: input y-value for point b
    :param x: x-value for the searched answer f_x
    :return: y-value for given x-value
    """
    if a == b == x and f_a == f_b:
        f_x = f_a
    else:
        slope = (f_b - f_a) / (b - a)
        f_x = slope * (x - a) + f_a
    return f_x


def interpolate_neighbours_linear(temperature, frequency, neighbours):
    """
    Linear interpolation of frequency and temperature between neighbours.

    :param temperature: desired temperature value in degree
    :param frequency: desired frequency value in Hz
    :param neighbours: neighbours
    :return: amplitude of the permittivity, angle of the permittivity
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
def create_permeability_measurement_in_database(material_name, measurement_setup, company="", date="", test_setup_name="", toroid_dimensions="",
                                                measurement_method="", equipment_names="", comment=""):
    """
    Create a new permeability section in the database for a material.

    :param material_name: name of the material
    :param measurement_setup: name of the measurement setup
    :param company: name of the company
    :param date: date of measurement
    :param test_setup_name: information of the test setup
    :param toroid_dimensions: dimensions of the probe
    :param measurement_method: name of the measurement method
    :param equipment_names: name of the measurement equipment
    :param comment: comment regarding the measurement
    :return: None
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


def clear_permeability_measurement_data_in_database(material_name, measurement_setup):
    """
    Clear the permeability data in the database given a material and measurement setup.

    :param material_name: name of the material
    :param measurement_setup: name of the measurement setup
    :return: None
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permeability_data_into_database(frequency, temperature, b_ref, mu_r_abs, mu_phi_deg, material_name, measurement_setup, overwrite=False):
    """
    Write permeability data into the material database.

    CAUTION: This method only adds the given measurement series to the permeability data without checking duplicates.

    :param temperature: temperature value in degree
    :param frequency: frequency value in Hz
    :param measurement_setup: name of the measurement setup
    :param b_ref: magnetic flux density value
    :param mu_r_abs: amplitude of the permeability
    :param mu_phi_deg: angle of the permeability
    :param material_name: name of the material
    :param overwrite: enable/disable overwritting of data
    :return: None
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
            "temperature": temperature,
            "frequency": frequency,
            "flux_density": list(b_ref),
            "mu_r_abs": list(mu_r_abs),
            "mu_phi_deg": list(mu_phi_deg)
        }
    )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# General Steinmetz --------------------------------------------------------------------------------------------------------------------------------------------
def write_steinmetz_data_into_database(temperature, k, beta, alpha, material_name, measurement_setup):
    """
    Write steinmetz data into the material database.

    CAUTION: This method only adds the given measurement series to the steinmetz data without checking duplicates.

    :param temperature: temperature value in degree
    :param k: k value of steinmetz parameters
    :param beta: beta value of the steinmetz parameters
    :param alpha: alpha value of the steinmetz parameters
    :param material_name: name of the material
    :param measurement_setup: name of the measurement setup
    :return: None
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


def create_empty_material(material_name: Material, manufacturer: Manufacturer):
    """
    Create an empty material slot in the database.

    :param material_name: name of the material
    :param manufacturer: name of the manufacturer
    :return: None
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if material_name in data:
        print(f"Material {material_name} already exists in materialdatabase.")
    else:
        data[material_name] = {
            "Manufacturer": manufacturer,
            "manufacturer_datasheet": {},
            "measurements": {}
        }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# General Permittivity -----------------------------------------------------------------------------------------------------------------------------------------
def create_permittivity_measurement_in_database(material_name, measurement_setup, company="", date="", test_setup_name="", probe_dimensions="",
                                                measurement_method="", equipment_names="", comment=""):
    """
    Create a new permittvity section in the database for a material.

    :param material_name: name of the material
    :param measurement_setup: name of the measurement setup
    :param company: name of the company
    :param date: date of measurement
    :param test_setup_name: information of the test setup
    :param probe_dimensions: dimensions of the probe
    :param measurement_method: name of the measurement method
    :param equipment_names: name of the measurement equipment
    :param comment: comment regarding the measurement
    :return: None
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


def clear_permittivity_measurement_data_in_database(material_name, measurement_setup):
    """
    Clear the permittivity data in the database for a specific material.

    :param material_name: name of material
    :param measurement_setup: name of measurement setup
    :return: None
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permittivity_data_into_database(temperature, frequencies, epsilon_r, epsilon_phi_deg, material_name, measurement_setup):
    """
    Write permittivity data into the material database.

    :param temperature: measurement point of the temperature in degree
    :param frequencies: measurement points of the frequency in Hz
    :param epsilon_r: amplitude of the permittivity
    :param epsilon_phi_deg: angle of the permittivity
    :param material_name: name of material
    :param measurement_setup: name of measurement setup
    :return: None
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
                "frequencies": frequencies,
                "epsilon_r": epsilon_r,
                "epsilon_phi_deg": epsilon_phi_deg
            }
        )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# LEA_LK Permeability ------------------------------------------------------------------------------------------------------------------------------------------
def get_permeability_data_from_lea_lk(location: str, frequency, temperature, material_name, no_interpolation_values: int = 20):
    """
    Get the permeability data from LEA_LK.

    :param location: location of the permeability data
    :param frequency: frequency value in Hz
    :param temperature: temperature value in degree
    :param material_name: name of the material
    :param no_interpolation_values: number of interpolation values
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


def create_permeability_file_name_lea_lk(quantity: str = "p_hys", frequency: int = 100000, material_name: str = "N49", temperature: int = 30):
    """
    Create the file name for permeability data of LEA_LK.

    :param quantity: measured quantiy (e.g. p_hys)
    :param frequency: frequency value in Hz
    :param material_name: name of the material
    :param temperature: temperature value in degree
    :return: correct file name for LEA_LK
    """
    return "_".join([quantity, f"{int(frequency / 1000)}kHz", material_name, f"{temperature}C.txt"])


def get_permeability_property_from_lea_lk(path_to_parent_folder, quantity: str, frequency: int, material_name: str, temperature: int,
                                          sub_folder_name: str = "Core_Loss"):
    """
    Get the proberty of the permeability from LEA_LK.

    :param path_to_parent_folder: path to permeability data
    :param quantity: name of the measured quantity
    :param frequency: frequency value in Hz
    :param material_name: name of the material
    :param temperature: temperature value in degree
    :param sub_folder_name: name of the sub folder
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
def get_permittivity_data_from_lea_lk(location, temperature, frequency, material_name):
    """
    Get the permittivity data from LEA_LK.

    :param location: location of the permittivity data
    :param temperature: temperature value
    :param frequency: frequency value in Hz
    :param material_name: name of the material
    :return: amplitude of the permittivity, angle of the permittivity
    """
    e_amplitude, epsilon_r_tilde = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_r_Plot", quantity="eps_r_tilde",
                                                                         frequency=frequency, material_name=material_name, temperature=temperature)

    e_phi, epsilon_phi_deg = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_phi_Plot", quantity="eps_phi_tilde",
                                                                   frequency=frequency, material_name=material_name, temperature=temperature)

    return epsilon_r_tilde, epsilon_phi_deg


def create_permittivity_file_name_lea_lk(quantity: str = "p_hys", frequency: int = 100000, material_name: str = "N49", temperature: int = 30):
    """
    Create the file name for permittivity data of LEA_LK.

    :param quantity: measured quantiy (e.g. p_hys)
    :param frequency: frequency value in Hz
    :param material_name: name of the material
    :param temperature: temperature value in degree
    :return: correct file name for LEA_LK
    """
    return "_".join([quantity, material_name, f"{temperature}C", f"{int(frequency / 1000)}kHz.txt"])


def get_permittivity_property_from_lea_lk(path_to_parent_folder, quantity: str, frequency: int, material_name: str, temperature: int,
                                          sub_folder_name: str = "Core_Loss"):
    """
    Get the proberty of the permittivity from LEA_LK.

    :param path_to_parent_folder: path to permittivity data
    :param quantity: name of the measured quantity
    :param frequency: frequency value in Hz
    :param material_name: name of the material
    :param temperature: temperature value in degree
    :param sub_folder_name: name of the sub folder
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
def get_permeability_property_from_lea_mtb(path_to_parent_folder):
    """
    Get the proberty of the permeability from the material test bench.

    :param path_to_parent_folder: path to permeability data
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
    :return: magnetic flux density, amplitude of the permeability, angle of the permeability
    """
    b_hys, mu_r_abs, mu_phi_deg = get_permeability_property_from_lea_mtb(path_to_parent_folder=location)

    return b_hys, mu_r_abs, mu_phi_deg


def get_all_frequencies_for_material(material_path):
    """
    Get all the frequency values for a given material.

    :param material_path: path to the material
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


def get_all_temperatures_for_directory(toroid_path):
    """
    Get all the temperature values for a given toroid probe.

    :param toroid_path: path of the toroid probe
    :return: all temperature values in degree of the specific toroid probe
    """
    temperatures_str = os.listdir(toroid_path)
    temperatures = []
    for f_str in temperatures_str:
        try:
            temperatures.append(int(f_str))
        except:
            pass
    return temperatures


def sigma_from_permittivity(amplitude_relative_equivalent_permittivity, phi_deg_relative_equivalent_permittivity, frequency):
    """
    Calculate the conductivity based on the data of the permittivity.

    :param amplitude_relative_equivalent_permittivity: amplitude of the permittivity
    :param phi_deg_relative_equivalent_permittivity: angle of the permittivity
    :param frequency: frequency value in Hz
    :return: conductivity
    """
    return 2 * np.pi * frequency * amplitude_relative_equivalent_permittivity * epsilon_0 * j * \
        (np.cos(np.deg2rad(phi_deg_relative_equivalent_permittivity)) + j * np.sin(np.deg2rad(phi_deg_relative_equivalent_permittivity)))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


# unused or externally used ------------------------------------------------------------------------------------------------------------------------------------
def find_nearest_frequencies(permeability, frequency):
    """
    Find the nearest frequency value for permeability data.

    :param permeability: permeability data
    :param frequency: desired frequency value in Hz
    :return: two frequency values in Hz with the desired value in between
    """
    freq_list = []
    # mdb_print(f"{freq_list = }")
    for j in range(len(permeability)):
        freq_list.append(permeability[j]["frequency"])
    # mdb_print(f"{freq_list = }")

    freq_list = list(remove(freq_list, len(freq_list)))
    # mdb_print(f"{freq_list = }")

    result = find_nearest(freq_list, frequency)

    return result[0], result[1]


def find_nearest_temperatures(permeability, f_l, f_h, temperature):
    """
    Find the nearest temperature value between two frequency points.

    :param permeability: permeability data
    :param f_l: lower frequency value in Hz
    :param f_h: higher frequency value in Hz
    :param temperature: desired temperature value in degree
    :return: two temperature values in degree with the desired value in between
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


def getdata_measurements(permeability, variable, frequency, temperature_1, temperature_2, b_t):
    """
    Linear interpolation of the permeability data between two temperatures at a constant frequency.

    :param permeability: permeability data
    :param variable: desired temperature variable in degree
    :param frequency: frequency value in Hz
    :param temperature_1: temperature value under the desired value in degree
    :param temperature_2: temperature value above the desired value in degree
    :param b_t: magnetic flux density
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


def export_data(parent_directory: str = "", file_format: str = None, b_ref_vec: list = None, mu_r_real_vec: list = None, mu_r_imag_vec: list = None,
                silent: bool = False):
    """
    Export data from the material database in a certain file format.

    :param parent_directory:
    :param b_ref_vec: reference vector for mu_r_real and mu_r_imag
    :param mu_r_imag_vec: imaginary part of mu_r_abs as a vector
    :param mu_r_real_vec: real part of mu_r_abs as a vector
    :param file_format: export format, e.g. 'pro' to export a .pro-file
    :param silent: enables/disables print
    :parent_directory:
    """
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


def plot_data(material_name: str = None, properties: str = None, b_ref: list = None, mu_r_real=None, mu_r_imag: list = None):
    """
    Plot certain material properties of materials.

    TODO: parameter is new and will probably cause problems when plotting data, but previous implementation was very static...
    :param b_ref: magnetic flux density value
    :param properties: name of the material properties
    :param material_name: name of the material
    :param mu_r_real: real part of the permeability
    :param mu_r_imag: imaginary part of the permeability
    :return: None
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
