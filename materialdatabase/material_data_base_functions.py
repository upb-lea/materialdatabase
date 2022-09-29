# all static functions shall be inserted in this file
import numpy as np
import json
import os
from scipy.interpolate import interp1d


# ------Remove Duplicate from freq array------
def remove(arr, n):
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


# -----find nearby frequency n Temp---------
def find_nearest(array, value):
    array = np.asarray(array)
    array.sort()
    idx = (np.abs(array - value)).argmin()
    if array[idx] > value:
        return array[idx - 1], array[idx]
    else:
        return array[idx], array[idx + 1]


# --------to get different material property from database file---------
def get_material_property(material_name: str, property: str):
    """

        :param material_name: str: N95,N87.....
        :param property: str:  initial_permeability, resistivity, max_flux_density, weight_density
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        r_data = json.load(data)
    value = r_data[f"{material_name}"]["manufacturer_datasheet"][f"{property}"]
    print(value)
    return value


# ----------to get steinmetz data from database file-----------------------
def get_steinmetz_data(material_name: str, type: str, datasource: str):
    """

    :param material_name:
    :param datasource: measurement or datasheet
    :param type: steinmetz or generalized steinmetz
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        s_data = json.load(data)

    s_data_new = s_data[f"{material_name}"][f"{datasource}"]
    if type == "Steinmetz":
        for i in range(len(s_data_new)):
            if s_data_new[i]["data_type"] == "steinmetz_data":
                coefficient = dict(s_data_new[i]["data"])
    # elif type == "Generalized_Steinmetz":
    #     coefficient = dict(s_data[f"{material_name}"]["generalized_steinmetz_data"])
    # print(coefficient)
    return coefficient
