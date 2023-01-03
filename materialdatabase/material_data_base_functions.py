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

# ------Remove Duplicate from freq array------
def remove(arr, n):
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


def rect(r, theta_deg):
    """
    converts polar coordinates [degree] kartesian coordinates
    theta in degrees
    :param r: radius or amplitude
    :param theta_deg: angle in degree

    :returns: tuple; (float, float); (x,y)
    """
    x = r * np.cos(np.radians(theta_deg))
    y = r * np.sin(np.radians(theta_deg))
    return x, y


# Permeability
def check_input_permeability_data(datasource, material_name, T, f):
    # mdb_print(datasource)
    if datasource != MaterialDataSource.Measurement and datasource != MaterialDataSource.ManufacturerDatasheet:
        raise Exception("'datasource' must be 'manufacturer_datasheet' or 'measurements'.")

    if material_name is None or T is None or f is None:
        raise Exception(f"Failure in selecting data from materialdatabase. {material_name = }, {T = }, {f =}.")


def getdata_datasheet(permeability, variable, F, t_1, t_2):
    for k in range(len(permeability)):
        if permeability[k]["frequency"] == F and permeability[k]["temperature"] == t_1:
            b_1 = permeability[k]["b"]
            mu_real_1 = permeability[k]["mu_real"]
            mu_imag_1 = permeability[k]["mu_imag"]
            t_mu_imag_1 = interp1d(b_1, mu_imag_1)
            t_mu_real_1 = interp1d(b_1, mu_real_1)
        if permeability[k]["frequency"] == F and permeability[k]["temperature"] == t_2:
            b_2 = permeability[k]["b"]
            mu_real_2 = permeability[k]["mu_real"]
            mu_imag_2 = permeability[k]["mu_imag"]
            t_mu_imag_2 = interp1d(b_2, mu_imag_2)
            t_mu_real_2 = interp1d(b_2, mu_real_2)

    # --------linear interpolation at constant freq-------------
    mu_i = []
    mu_r = []

    for y in range(len(b_1)):
        mu_r.append(t_mu_real_1(b_1[y]) + (t_mu_real_2(b_1[y]) - t_mu_real_1(b_1[y])) / (
                t_2 - t_1) * (variable - t_1))
        mu_i.append(t_mu_imag_1(b_1[y]) + (t_mu_imag_2(b_1[y]) - t_mu_imag_1(b_1[y])) / (
                t_2 - t_1) * (variable - t_1))
    return b_1, mu_r, mu_i


def create_permeability_neighbourhood_datasheet(T, f, list_of_permeability_dicts):
    """

    :param T:
    :param f:
    :param list_of_permeability_dicts:
    :return:
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_real": None,
                "mu_imag": None
            },
        "T_low_f_high":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_real": None,
                "mu_imag": None
            },
        "T_high_f_low":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_real": None,
                "mu_imag": None
            },
        "T_high_f_high":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_real": None,
                "mu_imag": None
            }
    }

    # In permeability data: find values of nearest neighbours
    T_value_low, T_value_high, f_value_low, f_value_high = find_nearest_neighbour_values_permeability(list_of_permeability_dicts, T, f)

    nbh["T_low_f_low"]["T"], nbh["T_low_f_high"]["T"] = T_value_low, T_value_low
    nbh["T_high_f_low"]["T"], nbh["T_high_f_high"]["T"] = T_value_high, T_value_high
    nbh["T_low_f_low"]["f"], nbh["T_high_f_low"]["f"] = f_value_low, f_value_low
    nbh["T_low_f_high"]["f"], nbh["T_high_f_high"]["f"] = f_value_high, f_value_high

    # find the indices of the neighbours in the original unsorted data
    for k_original, permeability_set in enumerate(list_of_permeability_dicts):

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_low:
            nbh["T_low_f_low"]["index"] = k_original
            nbh["T_low_f_low"]["b"] = permeability_set["b"]
            nbh["T_low_f_low"]["mu_real"] = permeability_set["mu_real"]
            nbh["T_low_f_low"]["mu_imag"] = permeability_set["mu_imag"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_low:
            nbh["T_high_f_low"]["index"] = k_original
            nbh["T_high_f_low"]["b"] = permeability_set["b"]
            nbh["T_high_f_low"]["mu_real"] = permeability_set["mu_real"]
            nbh["T_high_f_low"]["mu_imag"] = permeability_set["mu_imag"]

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_high:
            nbh["T_low_f_high"]["index"] = k_original
            nbh["T_low_f_high"]["b"] = permeability_set["b"]
            nbh["T_low_f_high"]["mu_real"] = permeability_set["mu_real"]
            nbh["T_low_f_high"]["mu_imag"] = permeability_set["mu_imag"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_high:
            nbh["T_high_f_high"]["index"] = k_original
            nbh["T_high_f_high"]["b"] = permeability_set["b"]
            nbh["T_high_f_high"]["mu_real"] = permeability_set["mu_real"]
            nbh["T_high_f_high"]["mu_imag"] = permeability_set["mu_imag"]

    return nbh


def create_permeability_neighbourhood_measurement(T, f, list_of_permeability_dicts):
    """

    :param T:
    :param f:
    :param list_of_permeability_dicts:
    :return:
    """
    # Initialize dicts for the certain operation point its neighbourhood
    nbh = {
        "T_low_f_low":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_r": None,
                "mu_phi_deg": None
            },
        "T_low_f_high":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_r": None,
                "mu_phi_deg": None
            },
        "T_high_f_low":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_r": None,
                "mu_phi_deg": None
            },
        "T_high_f_high":
            {
                "index": None,
                "T": None,
                "f": None,
                "b": None,
                "mu_r": None,
                "mu_phi_deg": None
            }
    }

    # In permeability data: find values of nearest neighbours
    T_value_low, T_value_high, f_value_low, f_value_high = find_nearest_neighbour_values_permeability(list_of_permeability_dicts, T, f)

    nbh["T_low_f_low"]["T"], nbh["T_low_f_high"]["T"] = T_value_low, T_value_low
    nbh["T_high_f_low"]["T"], nbh["T_high_f_high"]["T"] = T_value_high, T_value_high
    nbh["T_low_f_low"]["f"], nbh["T_high_f_low"]["f"] = f_value_low, f_value_low
    nbh["T_low_f_high"]["f"], nbh["T_high_f_high"]["f"] = f_value_high, f_value_high

    # find the indices of the neighbours in the original unsorted data
    for k_original, permeability_set in enumerate(list_of_permeability_dicts):

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_low:
            nbh["T_low_f_low"]["index"] = k_original
            nbh["T_low_f_low"]["b"] = permeability_set["b"]
            nbh["T_low_f_low"]["mu_r"] = permeability_set["mu_r"]
            nbh["T_low_f_low"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_low:
            nbh["T_high_f_low"]["index"] = k_original
            nbh["T_high_f_low"]["b"] = permeability_set["b"]
            nbh["T_high_f_low"]["mu_r"] = permeability_set["mu_r"]
            nbh["T_high_f_low"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_low and permeability_set["frequency"] == f_value_high:
            nbh["T_low_f_high"]["index"] = k_original
            nbh["T_low_f_high"]["b"] = permeability_set["b"]
            nbh["T_low_f_high"]["mu_r"] = permeability_set["mu_r"]
            nbh["T_low_f_high"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

        if permeability_set["temperature"] == T_value_high and permeability_set["frequency"] == f_value_high:
            nbh["T_high_f_high"]["index"] = k_original
            nbh["T_high_f_high"]["b"] = permeability_set["b"]
            nbh["T_high_f_high"]["mu_r"] = permeability_set["mu_r"]
            nbh["T_high_f_high"]["mu_phi_deg"] = permeability_set["mu_phi_deg"]

    return nbh


def find_nearest_neighbour_values_permeability(permeability_data, T, f):
    temperatures = []
    frequencies = []
    for permeability_set in permeability_data:
        temperatures.append(permeability_set["temperature"])
        frequencies.append(permeability_set["frequency"])

    # use sorted data without duplicates to find neighbours of operating point
    temperatures_sorted_without_duplicates = sorted(set(temperatures))
    frequencies_sorted_without_duplicates = sorted(set(frequencies))

    T_index_sorted_low, T_value_low, T_index_sorted_high, T_value_high = find_nearest_neighbours(T, temperatures_sorted_without_duplicates)
    f_index_sorted_low, f_value_low, f_index_sorted_high, f_value_high = find_nearest_neighbours(f, frequencies_sorted_without_duplicates)

    return T_value_low, T_value_high, f_value_low, f_value_high


def interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f, T_low, T_high, f_low, f_high,
                                                                  b_T_low_f_low, f_b_T_low_f_low,
                                                                  b_T_high_f_low, f_b_T_high_f_low,
                                                                  b_T_low_f_high, f_b_T_low_f_high,
                                                                  b_T_high_f_high, f_b_T_high_f_high,
                                                                  no_interpolation_values: int = 8, plot: bool = True):
    """

    :param T:
    :param f:
    :param T_low:
    :param T_high:
    :param f_low:
    :param f_high:
    :param b_T_low_f_low:
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
    if len(b_T_low_f_low) != len(f_b_T_low_f_low):
        raise ValueError(f"b_T_low_f_low and f_b_T_low_f_low must have the same lengths: \n"
                         f"is {len(b_T_low_f_low), len(f_b_T_low_f_low)}")

    # Interpolate functions of input data
    f_T_low_f_low_interpol = interp1d(b_T_low_f_low, f_b_T_low_f_low)
    f_T_high_f_low_interpol = interp1d(b_T_high_f_low, f_b_T_high_f_low)
    f_T_low_f_high_interpol = interp1d(b_T_low_f_high, f_b_T_low_f_high)
    f_T_high_f_high_interpol = interp1d(b_T_high_f_high, f_b_T_high_f_high)
    # mdb_print(f_T_low_f_low_interpol(1.5))

    # Find the border of the common magnetic flux density values
    b_max_min = max(min(b_T_low_f_low), min(b_T_high_f_low), min(b_T_low_f_high), min(b_T_high_f_high))
    b_min_max = min(max(b_T_low_f_low), max(b_T_high_f_low), max(b_T_low_f_high), max(b_T_high_f_high))
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
    if T_high == T_low:
        f_T_f_low_common = f_T_low_f_low_common  # at f_low
        f_T_f_high_common = f_T_low_f_high_common  # at f_high
    else:
        f_T_f_low_common = f_T_low_f_low_common + (f_T_high_f_low_common - f_T_low_f_low_common) / (T_high - T_low) * (T - T_low)  # at f_low
        f_T_f_high_common = f_T_low_f_high_common + (f_T_high_f_high_common - f_T_low_f_high_common) / (T_high - T_low) * (T - T_low)  # at f_high

    # Second interpolate in frequency:
    # mdb_print(f"{f_high, f_low = }")
    if f_low == f_high:
        f_T_f_common = f_T_f_low_common
    else:
        f_T_f_common = f_T_f_low_common + (f_T_f_high_common - f_T_f_low_common) / (f_high - f_low) * (f - f_low)

    if plot:
        plt.plot(b_common, f_T_low_f_low_common, linestyle='dashed', color="tab:blue", label=f"{T_low, f_low}")
        plt.plot(b_common, f_T_high_f_low_common, linestyle='dotted', color="tab:blue", label=f"{T_high, f_low}")
        plt.plot(b_common, f_T_low_f_high_common, linestyle='dashed', color="tab:red", label=f"{T_low, f_high}")
        plt.plot(b_common, f_T_high_f_high_common, linestyle='dotted', color="tab:red", label=f"{T_high, f_high}")
        plt.plot(b_common, f_T_f_low_common, color="tab:blue", label=f"{T, f_low}")
        plt.plot(b_common, f_T_f_high_common, color="tab:red", label=f"{T, f_high}")
        plt.plot(b_common, f_T_f_common, color="tab:orange", label=f"{T, f}")
        plt.legend()
        plt.grid()
        plt.show()

    return b_common, f_T_f_common


def mu_r__from_p_hyst_and_mu_phi_deg(mu_phi_deg, f, b_peak, p_hyst):
    """

    :param mu_phi_deg:
    :param f:
    :param b_peak:
    :param p_hyst:
    :return:
    """
    b_peak = np.array(b_peak)
    return b_peak ** 2 * np.pi * f * np.sin(np.deg2rad(mu_phi_deg)) / p_hyst / mu_0


def get_property_from_LEA_LK(path_to_parent_folder, quantity: str, f: int,
                             material_name: str, T: int, sub_folder_name: str = "Core_Loss"):
    filename = create_file_name_LEA_LK(quantity, f, material_name, T)
    complete_path = os.path.join(path_to_parent_folder, sub_folder_name, filename)
    # mdb_print(complete_path)

    data = np.loadtxt(complete_path)
    # mdb_print(data)
    return data[:, 0], data[:, 1]


def create_file_name_LEA_LK(quantity: str = "p_hys", f: int = 100000, material_name: str = "N49", T: int = 30):
    return quantity + "_" + f"{int(f / 1000)}" + "kHz_" + material_name + "_" + f"{T}" + "C.txt"


def get_permeability_data_from_LEA_LK(location: str, f, T, material_name, no_interpolation_values: int = 10):
    b_hys, p_hys = get_property_from_LEA_LK(path_to_parent_folder=location, sub_folder_name="Core_Loss",
                                            quantity="p_hys", f=f, material_name=material_name, T=T)
    b_phi, mu_phi_deg = get_property_from_LEA_LK(path_to_parent_folder=location, sub_folder_name="mu_phi_Plot",
                                                 quantity="mu_phi", f=f, material_name=material_name, T=T)

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

    return b_common, mu_r__from_p_hyst_and_mu_phi_deg(f_b_phi_interpol_common, f, b_common, f_p_hys_interpol_common), f_b_phi_interpol_common


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


# Permittivity
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


# unused or externally used

def find_nearest_frequencies(permeability, f):
    freq_list = []
    # mdb_print(f"{freq_list = }")
    for j in range(len(permeability)):
        freq_list.append(permeability[j]["frequency"])
    # mdb_print(f"{freq_list = }")

    freq_list = list(remove(freq_list, len(freq_list)))
    # mdb_print(f"{freq_list = }")

    result = find_nearest(freq_list, f)

    return result[0], result[1]


def find_nearest_temperatures(permeability, f_l, f_h, T):
    # ------find nearby temperature------
    temp_list_l = []
    temp_list_h = []

    for i in range(len(permeability)):
        if permeability[i]["frequency"] == f_l:
            temp_list_l.append(permeability[i]["temperature"])
    for i in range(len(permeability)):
        if permeability[i]["frequency"] == f_h:
            temp_list_h.append(permeability[i]["temperature"])

    return find_nearest(temp_list_l, T), find_nearest(temp_list_h, T)


def getdata_measurements(permeability, variable, F, t_1, t_2, b_t):
    for k in range(len(permeability)):
        if permeability[k]["frequency"] == F and permeability[k]["temperature"] == t_1:
            t_mu_phi_1 = interp1d(permeability[k]["b"], permeability[k]["mu_phi_deg"])
            t_mu_r_1 = interp1d(permeability[k]["mu_r"], permeability[k]["mu_r"])

        if permeability[k]["frequency"] == F and permeability[k]["temperature"] == t_2:
            t_mu_phi_2 = interp1d(permeability[k]["b"], permeability[k]["mu_phi_deg"])
            t_mu_r_2 = interp1d(permeability[k]["mu_r"], permeability[k]["mu_r"])
    # --------linear interpolation at constant freq-------------
    mu_phi = []
    mu_r = []

    for y in range(len(b_t)):
        mu_r.append(t_mu_r_1(b_t[y]) + (t_mu_r_2(b_t[y]) - t_mu_r_1(b_t[y])) / (t_2 - t_1) * (variable - t_1))
        mu_phi.append(t_mu_phi_1(b_t[y]) + (t_mu_phi_2(b_t[y]) - t_mu_phi_1(b_t[y])) / (t_2 - t_1) * (variable - t_1))
    return mu_r, mu_phi


def export_data(parent_directory: str = "", file_format: str = None,
                b_ref: list = None, mu_real: list = None, mu_imag: list = None):
    """
    Method is used to export data from the material database in a certain file format.
    :param b_ref:
    :param mu_imag:
    :param mu_real:
    :param file_format: export format
    :parent_directory:
    @param file_format:
    @param parent_directory:
    """
    if file_format == "pro":
        with open(os.path.join(parent_directory, "core_materials_temp.pro"), "w") as file:
            file.write(f'Include "Parameter.pro";\n')
            file.write(
                f"Function{{\n  b = {str(b_ref).replace('[', '{').replace(']', '}')} ;\n  mu_real = {str(mu_real).replace('[', '{').replace(']', '}')} ;"
                f"\n  mu_imag = {str(mu_imag).replace('[', '{').replace(']', '}')} ;\n  "
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
              b_ref: list = None, mu_real=None, mu_imag: list = None):
    """
    Method is used to plot certain material properties of materials.
    :param b_ref: TODO: parameter is new and will probably cause problems when plotting data, but previous implementation was very static...
    :param properties:
    :param material_name:
    :return:
    """
    if properties == "mu_real":
        plt.plot(b_ref, mu_real)
        plt.ylabel(properties)
        plt.xlabel('B in T')
        plt.title("Real part of permeability")
        plt.show()
    elif properties == "mu_imag":
        plt.plot(b_ref, mu_imag)
        plt.ylabel(properties)
        plt.xlabel('B in T')
        plt.title("Imaginary part of permeability")
        plt.show()

    mdb_print(f"Material properties {properties} of {material_name} are plotted.")


def clear_permeability_data_in_database(material_name, measurement_setup):
    """

    :param material_name:
    :param measurement_setup:
    :return:
    """
    relative_path_to_db = "../data/material_data_base.json"
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["permeability_data"] = []

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def write_permeability_data_into_database(f, T, b_ref, mu_r, mu_phi_deg, material_name, measurement_setup):
    """
    CAUTION: This method only adds the given measurement series to the permeability data
    without checking duplicates!
    :param measurement_setup:
    :param b_ref:
    :param mu_r:
    :param mu_phi_deg:
    :param material_name:
    :return:
    """
    relative_path_to_db = "../data/material_data_base.json"
    with open(relative_path_to_db, "r") as jsonFile:
        data = json.load(jsonFile)

    if type(data[material_name]["measurements"]["complex_permeability"][measurement_setup]["permeability_data"]) is not list:
        data[material_name]["measurements"]["complex_permeability"][measurement_setup]["permeability_data"] = []

    data[material_name]["measurements"]["complex_permeability"][measurement_setup]["permeability_data"].append(
        {
            "temperature": T,
            "frequency": f,
            "b": list(b_ref),
            "mu_r": list(mu_r),
            "mu_phi_deg": list(mu_phi_deg)
        }
    )

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
