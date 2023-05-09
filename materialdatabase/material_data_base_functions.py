# all static functions shall be inserted in this file

# Python integrated libraries

import json
# 3rd party libraries
import os

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as savgol

# local libraries
from .constants import *
from .enumerations import *

# Relative path to the database json file
global relative_path_to_db
relative_path_to_db = "../data/material_data_base.json"


# ---
# Auxiliary functions

def remove(arr, n):
    """
    Remove Duplicate from array
    :param arr:
    :param n:
    :return:
    """
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


def crop_data_fixed(x, pre_cropped_values: int = 0, post_cropped_values: int = 0):
    return x[pre_cropped_values:-post_cropped_values]


def crop_data_variable_length(x):
    pre_cropped_values = 0
    post_cropped_values = 0

    # determine no of pre crop data
    rolling_average = []
    relative_rolling_average_change = 0
    while relative_rolling_average_change < 1:
        relative_rolling_average_change


def store_data(material_name: str, data_to_be_stored: dict) -> None:
    """
    Method is used to store data from measurement/datasheet into the material database.
    :param material_name: Material name
    :type material_name: str
    :param data_to_be_stored: data to be stored
    :type data_to_be_stored: dict
    :return: None
    :rtype: None
    """
    with open('material_data_base.json', 'w') as outfile:
        json.dump(data_to_be_stored, outfile, indent=4)
    mdb_print(f"Material properties of {material_name} are stored in the material database.")


def find_nearest(array, value):
    """
    find nearby frequency n Temp
    :param array:
    :param value:
    :return:
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


def set_silent_status(is_silent: bool) -> None:
    """
    Silent mode global variable.

    :param is_silent: True for silent mode, False for mode with print outputs
    :type is_silent: bool
    """
    global silent
    silent = is_silent


def mdb_print(text: str, end='\n') -> None:
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


def rect(radius_or_amplitude: float, theta_deg: float):
    """
    converts polar coordinates [degree] cartesian coordinates
    theta in degrees
    :param radius_or_amplitude: radius or amplitude
    :type radius_or_amplitude: float
    :param theta_deg: angle in degree
    :type theta_deg: float

    :returns: tuple; (float, float); (abscissa_x,ordinate_y)
    """
    abscissa_x = radius_or_amplitude * np.cos(np.radians(theta_deg))
    ordinate_y = radius_or_amplitude * np.sin(np.radians(theta_deg))
    return abscissa_x, ordinate_y

# ---
# Load Permeability

def check_input_permeability_data(datasource: str, material_name: str, temperature: float, frequency: float) -> None:
    """
    Checks input permeability data for correct input parameters.
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
    :returns: None
    :rtype: None
    """
    if datasource != MaterialDataSource.Measurement and datasource != MaterialDataSource.ManufacturerDatasheet:
        raise Exception("'datasource' must be 'manufacturer_datasheet' or 'measurements'.")

    if material_name is None or temperature is None or frequency is None:
        raise Exception(f"Failure in selecting data from materialdatabase. {material_name = }, {temperature = }, {frequency =}.")


def getdata_datasheet(permeability, variable, frequency, temperature_1, temperature_2):
    """
    interpolation function
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
        mu_r.append(t_mu_real_1(b_1[y]) + (t_mu_real_2(b_1[y]) - t_mu_real_1(b_1[y])) / (
                temperature_2 - temperature_1) * (variable - temperature_1))
        mu_i.append(t_mu_imag_1(b_1[y]) + (t_mu_imag_2(b_1[y]) - t_mu_imag_1(b_1[y])) / (
                temperature_2 - temperature_1) * (variable - temperature_1))
    return b_1, mu_r, mu_i


def create_permeability_neighbourhood_datasheet(temperature, frequency, list_of_permeability_dicts):
    """

    :param temperature:
    :param frequency:
    :param list_of_permeability_dicts:
    :return:
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

    :param temperature:
    :param frequency:
    :param list_of_permeability_dicts:
    :return:
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
                                                                  b_t_low_f_low, f_b_T_low_f_low,
                                                                  b_T_high_f_low, f_b_T_high_f_low,
                                                                  b_T_low_f_high, f_b_T_low_f_high,
                                                                  b_T_high_f_high, f_b_T_high_f_high,
                                                                  no_interpolation_values: int = 8,
                                                                  y_label: str = None, plot: bool = False):
    """

    :param y_label:
    :param temperature:
    :param frequency:b_ref_vec
    :param temperature_low:
    :param temperature_high:
    :param frequency_low:
    :param frequency_high:
    :param b_t_low_f_low:
    :param f_b_T_low_f_low:
    :param b_T_high_f_low:
    :param f_b_T_high_f_low:
    :param b_T_low_f_high:
    :param f_b_T_low_f_high:
    :param b_T_high_f_high:
    :param f_b_T_high_f_high:
    :param no_interpolation_values:
    :param plot:
    :return:
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
        f_T_f_low_common = f_T_low_f_low_common + (f_T_high_f_low_common - f_T_low_f_low_common) / (temperature_high - temperature_low) * (temperature - temperature_low)  # at f_low
        f_T_f_high_common = f_T_low_f_high_common + (f_T_high_f_high_common - f_T_low_f_high_common) / (temperature_high - temperature_low) * (temperature - temperature_low)  # at f_high

    # Second interpolate in frequency:
    # mdb_print(f"{f_high, f_low = }")
    if frequency_low == frequency_high:
        f_T_f_common = f_T_f_low_common
    else:
        f_T_f_common = f_T_f_low_common + (f_T_f_high_common - f_T_f_low_common) / (frequency_high - frequency_low) * (frequency - frequency_low)

    if plot:
        scale = 1000
        plt.plot(b_common * scale, f_T_low_f_low_common, linestyle='dashed', color="tab:blue", label=r"$T_\mathregular{low}$" + f"={temperature_low} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_low_f_high_common, linestyle='dashed', color="tab:red", label=r"$T_\mathregular{low}$" + f"={temperature_low} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")

        plt.plot(b_common * scale, f_T_high_f_low_common, linestyle='dotted', color="tab:blue", label=r"$T_\mathregular{high}$" + f"={temperature_high} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_high_f_high_common, linestyle='dotted', color="tab:red", label=r"$T_\mathregular{high}$" + f"={temperature_high} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")

        plt.plot(b_common * scale, f_T_f_low_common, color="tab:blue", label=r"$T$" + f"={temperature} and " + r"$f_\mathregular{low}$" + f"={frequency_low}")
        plt.plot(b_common * scale, f_T_f_high_common, color="tab:red", label=r"$T$" + f"={temperature} and " + r"$f_\mathregular{high}$" + f"={frequency_high}")
        plt.plot(b_common * scale, f_T_f_common, color="tab:orange", label=r"$T$" + f"={temperature} and " + r"$f$" + f"={frequency}")
        plt.xlabel("amplitude of magnetic flux density in mT")
        plt.ylabel(f"{y_label}")
        plt.title(f"Interpolation in temperature and frequency")
        plt.legend()
        plt.grid()
        plt.show()

    return b_common, f_T_f_common


def mu_r__from_p_hyst_and_mu_phi_deg(mu_phi_deg, frequency, b_peak, p_hyst):
    """

    :param mu_phi_deg:
    :param frequency: frequency
    :param b_peak: peak flux density
    :param p_hyst: hysteresis losses
    :return:
    """
    b_peak = np.array(b_peak)
    return b_peak ** 2 * np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) / p_hyst / mu_0


def p_hyst__from_mu_r_and_mu_phi_deg(frequency, b_peak, mu_r, mu_phi_deg):
    """

    :param mu_phi_deg:
    :param frequency: frequency
    :param b_peak: peak flux density
    :param p_hyst: hysteresis losses
    :return:
    """
    b_peak = np.array(b_peak)
    return np.pi * frequency * np.sin(np.deg2rad(mu_phi_deg)) * mu_0 * mu_r * (b_peak / mu_0 / mu_r) ** 2


def process_permeability_data(b_ref_raw, mu_r_raw, mu_phi_deg_raw,
                              smooth_data: bool = False, crop_data: bool = False,
                              plot_data: bool = False):
    """

    :param b_ref_raw:
    :param mu_r_raw:
    :param mu_phi_deg_raw:
    :param smooth_data:
    :param crop_data:
    :param plot_data:
    :return:
    """
    if crop_data:
        pre, end = 10, 5
        b_ref = crop_data_fixed(b_ref_raw, pre, end)
        mu_r = crop_data_fixed(mu_r_raw, pre, end)
        mu_phi_deg = crop_data_fixed(mu_phi_deg_raw, pre, end)
    else:
        b_ref, mu_r, mu_phi_deg = b_ref_raw, mu_r_raw, mu_phi_deg_raw

    if smooth_data:
        mu_r = savgol(x=mu_r, window_length=10, polyorder=2)
        mu_phi_deg = savgol(x=mu_phi_deg, window_length=10, polyorder=2)

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
        fig, ax = plt.subplots(2)

        # Amplitude Plot
        ax[0].plot(b_ref_raw, mu_r_raw)
        ax[0].plot(b_ref, mu_r)
        ax[0].grid()

        # Phase Plot
        ax[1].plot(b_ref_raw, mu_phi_deg_raw)
        if smooth_data:
            ax[1].plot(b_ref, mu_phi_deg)
        ax[1].grid()

        plt.show()

    return b_ref, mu_r, mu_phi_deg



# ---
# Load Permittivity

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
    
    :param temperature: 
    :param frequency: 
    :param list_of_permittivity_dicts: 
    :return: 
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
    index_T_low_neighbour, value_T_low_neighbour, index_T_high_neighbour, value_T_high_neighbour = \
        find_nearest_neighbours(temperature, temperatures)

    nbh["T_low_f_low"]["temperature"]["value"], nbh["T_low_f_high"]["temperature"]["value"] = value_T_low_neighbour, value_T_low_neighbour
    nbh["T_low_f_low"]["temperature"]["index"], nbh["T_low_f_high"]["temperature"]["index"] = index_T_low_neighbour, index_T_low_neighbour
    nbh["T_high_f_low"]["temperature"]["value"], nbh["T_high_f_high"]["temperature"]["value"] = value_T_high_neighbour, value_T_high_neighbour
    nbh["T_high_f_low"]["temperature"]["index"], nbh["T_high_f_high"]["temperature"]["index"] = index_T_high_neighbour, index_T_high_neighbour

    # T low
    nbh["T_low_f_low"]["frequency"]["index"], nbh["T_low_f_low"]["frequency"]["value"], \
    nbh["T_low_f_high"]["frequency"]["index"], nbh["T_low_f_high"]["frequency"]["value"] = \
        find_nearest_neighbours(frequency, list_of_permittivity_dicts[index_T_low_neighbour]["frequencies"])

    nbh["T_low_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["temperature"]["index"]]["epsilon_r"][nbh["T_low_f_low"]["frequency"]["index"]]
    nbh["T_low_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_low"]["temperature"]["index"]]["epsilon_phi_deg"][nbh["T_low_f_low"]["frequency"]["index"]]
    nbh["T_low_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["temperature"]["index"]]["epsilon_r"][nbh["T_low_f_high"]["frequency"]["index"]]
    nbh["T_low_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_low_f_high"]["temperature"]["index"]]["epsilon_phi_deg"][nbh["T_low_f_high"]["frequency"]["index"]]

    # T high
    nbh["T_high_f_low"]["frequency"]["index"], nbh["T_high_f_low"]["frequency"]["value"], \
    nbh["T_high_f_high"]["frequency"]["index"], nbh["T_high_f_high"]["frequency"]["value"] = \
        find_nearest_neighbours(frequency, list_of_permittivity_dicts[index_T_high_neighbour]["frequencies"])

    nbh["T_high_f_low"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["temperature"]["index"]]["epsilon_r"][nbh["T_high_f_low"]["frequency"]["index"]]
    nbh["T_high_f_low"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_low"]["temperature"]["index"]]["epsilon_phi_deg"][nbh["T_high_f_low"]["frequency"]["index"]]
    nbh["T_high_f_high"]["epsilon_r"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["temperature"]["index"]]["epsilon_r"][nbh["T_high_f_high"]["frequency"]["index"]]
    nbh["T_high_f_high"]["epsilon_phi_deg"] = list_of_permittivity_dicts[nbh["T_high_f_high"]["temperature"]["index"]]["epsilon_phi_deg"][nbh["T_high_f_high"]["frequency"]["index"]]

    return nbh


def my_interpolate_linear(a, b, f_a, f_b, x):
    """
    interpolates linear between to points 'a' and 'b'.
    The return value is f_x in dependence of x
    Tt applies: a < x < b.

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
    interpolates or extrapolates linear for a<x<b or x<a and x>b

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

    :param temperature:
    :param frequency:
    :param neighbours:
    :return:
    """
    # Interpolation of Amplitude
    # in temperature at f_low
    epsilon_r_at_T_f_low = my_polate_linear(a=neighbours["T_low_f_low"]["temperature"]["value"], b=neighbours["T_high_f_low"]["temperature"]["value"],
                                            f_a=neighbours["T_low_f_low"]["epsilon_r"], f_b=neighbours["T_high_f_low"]["epsilon_r"],
                                            x=temperature)
    # in temperature at f_high
    epsilon_r_at_T_f_high = my_polate_linear(a=neighbours["T_low_f_high"]["temperature"]["value"], b=neighbours["T_high_f_high"]["temperature"]["value"],
                                             f_a=neighbours["T_low_f_high"]["epsilon_r"], f_b=neighbours["T_high_f_high"]["epsilon_r"],
                                             x=temperature)
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


# ---
# Add and remove data in Database

# General
# Permeability
def create_permeability_measurement_in_database(material_name, measurement_setup, company="", date="", test_setup_name="",
                                                toroid_dimensions="", measurement_method="", equipment_names="", comment=""):
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permeability"] = {
        measurement_setup: {
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
    }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def clear_permeability_measurement_data_in_database(material_name, measurement_setup):
    """

    :param material_name:
    :param measurement_setup:
    :return:
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permeability_data_into_database(frequency, temperature, b_ref, mu_r_abs, mu_phi_deg, material_name, measurement_setup):
    """
    CAUTION: This method only adds the given measurement series to the permeability data
    without checking duplicates!
    :param temperature:
    :param frequency:
    :param measurement_setup:
    :param b_ref:
    :param mu_r_abs:
    :param mu_phi_deg:
    :param material_name:
    :return:
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if type(data[material_name]["measurements"]["complex_permeability"][measurement_setup]["measurement_data"]) is not list:
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


# General
# Permittivity
def create_permittivity_measurement_in_database(material_name, measurement_setup, company="", date="", test_setup_name="",
                                                probe_dimensions="", measurement_method="", equipment_names="", comment=""):
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permittivity"] = {
        measurement_setup: {
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
    }

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def clear_permittivity_measurement_data_in_database(material_name, measurement_setup):
    """

    :param material_name:
    :param measurement_setup:
    :return:
    """
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permittivity_data_into_database(temperature, frequencies, epsilon_r, epsilon_phi_deg, material_name, measurement_setup):
    # load data

    # mean of data

    # write data in DB
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if type(data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"]) is not list:
        data[material_name]["measurements"]["complex_permittivity"][measurement_setup]["measurement_data"] = []

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


# LEA_LK
# Permeability
def get_permeability_data_from_lea_lk(location: str, frequency, temperature, material_name, no_interpolation_values: int = 10):
    b_hys, p_hys = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="Core_Loss",
                                                         quantity="p_hys", frequency=frequency, material_name=material_name, temperature=temperature)
    b_phi, mu_phi_deg = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="mu_phi_Plot",
                                                              quantity="mu_phi", frequency=frequency, material_name=material_name, temperature=temperature)

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
    return "_".join([quantity, f"{int(frequency / 1000)}kHz", material_name, f"{temperature}C.txt"])


def get_permeability_property_from_lea_lk(path_to_parent_folder, quantity: str, frequency: int,
                                          material_name: str, temperature: int, sub_folder_name: str = "Core_Loss"):
    filename = create_permeability_file_name_lea_lk(quantity, frequency, material_name, temperature)
    complete_path = os.path.join(path_to_parent_folder, sub_folder_name, filename)
    # mdb_print(complete_path)

    data = np.loadtxt(complete_path)
    # mdb_print(data)
    return data[:, 0], data[:, 1]


# Permittivity
def get_permittivity_data_from_lea_lk(location, temperature, frequency, material_name):
    e_amplitude, epsilon_r_tilde = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_r_Plot",
                                                                         quantity="eps_r_tilde", frequency=frequency, material_name=material_name, temperature=temperature)

    e_phi, epsilon_phi_deg = get_permittivity_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="eps_phi_Plot",
                                                                   quantity="eps_phi_tilde", frequency=frequency, material_name=material_name, temperature=temperature)

    return epsilon_r_tilde, epsilon_phi_deg


def create_permittivity_file_name_lea_lk(quantity: str = "p_hys", frequency: int = 100000, material_name: str = "N49", temperature: int = 30):
    return "_".join([quantity, material_name, f"{temperature}C", f"{int(frequency / 1000)}kHz.txt"])



def get_permittivity_property_from_lea_lk(path_to_parent_folder, quantity: str, frequency: int,
                                          material_name: str, temperature: int, sub_folder_name: str = "Core_Loss"):
    filename = create_permittivity_file_name_lea_lk(quantity, frequency, material_name, temperature)
    complete_path = os.path.join(path_to_parent_folder, sub_folder_name, filename)
    # mdb_print(complete_path)

    data = np.loadtxt(complete_path)
    # mdb_print(data)
    return data[:, 0], data[:, 1]


# ---
# unused or externally used

def find_nearest_frequencies(permeability, frequency):
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


def export_data(parent_directory: str = "", file_format: str = None,
                b_ref_vec: list = None, mu_r_real_vec: list = None, mu_r_imag_vec: list = None):
    """
    Method is used to export data from the material database in a certain file format.

    :param b_ref_vec: reference vector for mu_r_real and mu_r_imag
    :param mu_r_imag_vec: imaginary part of mu_r_abs as a vector
    :param mu_r_real_vec: real part of mu_r_abs as a vector
    :param file_format: export format, e.g. 'pro' to export a .pro-file
    :parent_directory:
    """
    if file_format == "pro":
        with open(os.path.join(parent_directory, "core_materials_temp.pro"), "w") as file:
            file.write(f'Include "Parameter.pro";\n')
            file.write(
                f"Function{{\n  b = {str(b_ref_vec).replace('[', '{').replace(']', '}')} ;\n  mu_real = {str(mu_r_real_vec).replace('[', '{').replace(']', '}')} ;"
                f"\n  mu_imag = {str(mu_r_imag_vec).replace('[', '{').replace(']', '}')} ;\n  "
                f"mu_imag_couples = ListAlt[b(), mu_imag()] ;\n  "
                f"mu_real_couples = ListAlt[b(), mu_real()] ;\n  "
                f"f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{{List[mu_imag_couples]}};\n  "
                f"f_mu_real_d[] = InterpolationLinear[Norm[$1]]{{List[mu_real_couples]}};\n  "
                f"f_mu_imag[] = f_mu_imag_d[$1];\n  "
                f"f_mu_real[] = f_mu_real_d[$1];\n }}  ")

    else:
        raise Exception("No valid file format is given!")

    mdb_print(f"Data is exported to {parent_directory} in a {file_format}-file.")


def plot_data(material_name: str = None, properties: str = None,
              b_ref: list = None, mu_r_real=None, mu_r_imag: list = None):
    """
    Method is used to plot certain material properties of materials.
    :param b_ref: TODO: parameter is new and will probably cause problems when plotting data, but previous implementation was very static...
    :param properties:
    :param material_name:
    :return:
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

    mdb_print(f"Material properties {properties} of {material_name} are plotted.")
