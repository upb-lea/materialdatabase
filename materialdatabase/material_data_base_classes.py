import os

import femmt as fmt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from material_data_base_functions import *


class MaterialDatabase:
    """
    This class manages the data stored in the material database.
    It has the possibility to export for example soft magnetic materials' loss data in certain format,
    so that it can easily be interfaced by tools like the FEM Magnetic Toolbox.
    """

    def __init__(self):
        print("The material database is now initialized")
        self.freq = None
        self.temp = None
        self.mat = None
        self.mu_real = None
        self.mu_imag = None

    def get_data_at_working_point(self, T: int, f: int, material_name):
        """
        Method is used to read data from the material database.
        :param T:
        :param f:
        :param material_name:
        :return:
        """
        self.temp = T
        self.freq = f
        self.mat = material_name

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'data/material_data_base.json')
        with open(file_path, 'r') as database:
            m_data = json.load(database)
        freq_list = []
        # print(len(m_data["data"]))
        for i in range(len(m_data["data"])):
            freq_list.append(m_data["data"][i]["frequency"])
        # print(freq_list)
        n = len(freq_list)  # len of array
        freq_list = list(remove(freq_list, n))
        # print(freq_list)

        result = find_nearest(freq_list, f)
        print(result)

        f_l = result[0]
        f_h = result[1]

        # ------find nearby temperature------
        temp_list_l = []
        temp_list_h = []

        for i in range(len(m_data["data"])):
            if m_data["data"][i]["frequency"] == f_l:
                temp_list_l.append(m_data["data"][i]["temperature"])
        for i in range(len(m_data["data"])):
            if m_data["data"][i]["frequency"] == f_h:
                temp_list_h.append(m_data["data"][i]["temperature"])

        temp_list_l = find_nearest(temp_list_l, T)
        temp_list_h = find_nearest(temp_list_h, T)
        print(temp_list_l)
        # print(temp_list_h)

        # -------get the data----------
        def getdata(variable, F, t_1, t_2):
            global t_mu_imag_1, t_mu_imag_2, t_mu_real_2, t_mu_real_1
            for k in range(len(m_data["data"])):
                if m_data["data"][k]["frequency"] == F and m_data["data"][k]["temperature"] == t_1:
                    b_1 = m_data["data"][k]["b"]
                    mu_real_1 = m_data["data"][k]["mu_real"]
                    mu_imag_1 = m_data["data"][k]["mu_imag"]
                    t_mu_imag_1 = interp1d(b_1, mu_imag_1)
                    t_mu_real_1 = interp1d(b_1, mu_real_1)
                if m_data["data"][k]["frequency"] == F and m_data["data"][k]["temperature"] == t_2:
                    b_2 = m_data["data"][k]["b"]
                    mu_real_2 = m_data["data"][k]["mu_real"]
                    mu_imag_2 = m_data["data"][k]["mu_imag"]
                    t_mu_imag_2 = interp1d(b_2, mu_imag_2)
                    t_mu_real_2 = interp1d(b_2, mu_real_2)

            # --------linear interpolation at constant freq-------------
            mu_i = []
            mu_r = []
            b_f = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

            for j in range(len(b_f)):
                mu_r.append(
                    t_mu_real_1(b_f[j]) + (t_mu_real_2(b_f[j]) - t_mu_real_1(b_f[j])) / (t_2 - t_1) * (variable - t_1))
                mu_i.append(
                    t_mu_imag_1(b_f[j]) + (t_mu_imag_2(b_f[j]) - t_mu_imag_1(b_f[j])) / (t_2 - t_1) * (variable - t_1))
            return mu_r, mu_i

        # --------interpolated data at constant freq and nearby temp--------
        interpolate_temp_1 = getdata(T, f_l, temp_list_l[0], temp_list_l[1])
        interpolate_temp_2 = getdata(T, f_h, temp_list_h[0], temp_list_h[1])
        # print(interpolate_temp_1)
        # print(interpolate_temp_2)

        # ------linear interpolation at constant temp and nearby freq-----------------
        b_g = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        f_mu_real_1 = interp1d(b_g, interpolate_temp_1[0])
        f_mu_imag_1 = interp1d(b_g, interpolate_temp_1[1])
        f_mu_real_2 = interp1d(b_g, interpolate_temp_2[0])
        f_mu_imag_2 = interp1d(b_g, interpolate_temp_2[1])
        mu_i_f = []
        mu_r_f = []
        for b in range(len(b_g)):
            mu_r_f.append(
                f_mu_real_1(b_g[b]) + (f_mu_real_2(b_g[b]) - f_mu_real_1(b_g[b])) / (f_h - f_l) * (f - f_l))
            mu_i_f.append(
                f_mu_imag_1(b_g[b]) + (f_mu_imag_2(b_g[b]) - f_mu_imag_1(b_g[b])) / (f_h - f_l) * (f - f_l))
        self.mu_real = mu_r_f
        self.mu_imag = mu_i_f
        print(self.mu_real)
        print(self.mu_imag)

        print(f"Material properties of {material_name} are loaded at {T} °C and {f} Hz.")
        pass

    def export_data(self, data_to_export, format):
        """
        Method is used to export data from the material database in a certain file format.
        :param data_to_export:
        :param format:
        :return:
        """
        print(f"Data {data_to_export} is exported in a {format}-file.")
        pass

    def store_data(self, material_name, data_to_be_stored):
        """
        Method is used to store data from measurement/datasheet into the material database.
        :param material_name:
        :param data_to_be_stored:
        :return:
        """
        print(f"Material properties of {material_name} are stored in the material database.")
        pass

    def plot_data(self, material_name: str = None, properties: str = None):
        """
        Method is used to plot certain material properties of materials.
        :param properties:
        :param material_name:
        :return:
        """
        plt.plot([1, 2, 3], [10, 20, 30])
        plt.ylabel(properties)
        plt.xlabel('B in mT')
        plt.show()
        print(f"Material properties {properties} of {material_name} are plotted.")
        pass

    def compare_data(self, material_name_list):
        """
        Method is used to compare material properties.
        :param material_name_list:
        :return:
        """
        print(f"Material properties of {material_name_list} are compared.")
        pass
