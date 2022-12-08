# Python integrated libraries
import os

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import mplcursors

# local libraries
from materialdatabase.material_data_base_functions import *


class MaterialDatabase:
    """
    This class manages the data stored in the material database.
    It has the possibility to export for example soft magnetic materials' loss data in certain format,
    so that it can easily be interfaced by tools like the FEM Magnetic Toolbox.
    """

    def __init__(self, is_silent: bool = False):

        self.b_1 = None
        self.freq = None
        self.temp = None
        self.mat = None
        self.b_f = None
        self.mu_real = None
        self.mu_imag = None
        set_silent_status(is_silent)

        mdb_print("The material database is now initialized")

    def permeability_data_to_pro_file(self, T: float, f: float, material_name: str, datasource: str, pro: bool = False,
                                      parent_directory: str = ""):
        """
        Method is used to read permeability data from the material database.
        :param T: temperature
        :param f: Frequency
        :param material_name: "N95","N87"....
        :param datasource: "measurements" or "manufacturer_datasheet"
        :param pro : create temporary pro file for solver
        :param parent_directory: location of solver file
        """
        if material_name is None or T is None or f is None:
            raise Exception(f"Failure in selecting data from materialdatabase. {material_name = }, {T = }, {f =}.")

        self.temp = T
        self.freq = f
        self.mat = material_name

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'data/material_data_base.json')
        with open(file_path, 'r') as database:
            data = json.load(database)
        freq_list = []

        if datasource == "measurements":
            m_data = data[f"{material_name}"][f"{datasource}"]
            # print(len(m_data["permeability_data"]))
            for i in range(len(m_data)):
                if m_data[i]["data_type"] == "complex_permeability_data":
                    m_data_new = m_data[i]["permeability_data"]
            for j in range(len(m_data_new)):
                freq_list.append(m_data_new[j]["frequency"])
                # print(freq_list)
            n = len(freq_list)  # len of array
            freq_list = list(remove(freq_list, n))
            print(freq_list)

            result = find_nearest(freq_list, f)
            print(result)

            f_l = result[0]
            f_h = result[1]

            # ------find nearby temperature------
            temp_list_l = []
            temp_list_h = []

            for i in range(len(m_data_new)):
                if m_data_new[i]["frequency"] == f_l:
                    temp_list_l.append(m_data_new[i]["temperature"])
            for i in range(len(m_data_new)):
                if m_data_new[i]["frequency"] == f_h:
                    temp_list_h.append(m_data_new[i]["temperature"])

            temp_list_l = find_nearest(temp_list_l, T)
            temp_list_h = find_nearest(temp_list_h, T)

            # print(temp_list_l)
            # print(temp_list_h)

            def getdata_measurements(variable, F, t_1, t_2):
                for k in range(len(m_data_new)):
                    if m_data_new[k]["frequency"] == F and m_data_new[k]["temperature"] == t_1:
                        b_phi_deg_1 = m_data_new[k]["b"]
                        mu_phi_deg_1 = m_data_new[k]["mu_phi_deg"]
                        mu_r_1_and_b = m_data_new[k]["mu_r"]
                        b_r_1 = mu_r_1_and_b[0]
                        mu_r_1 = mu_r_1_and_b[1]
                        t_mu_phi_1 = interp1d(b_phi_deg_1, mu_phi_deg_1)
                        t_mu_r_1 = interp1d(b_r_1, mu_r_1)

                    if m_data_new[k]["frequency"] == F and m_data_new[k]["temperature"] == t_2:
                        b_phi_deg_2 = m_data_new[k]["b"]
                        mu_phi_deg_2 = m_data_new[k]["mu_phi_deg"]
                        mu_r_2_and_b = m_data_new[k]["mu_r"]
                        b_r_2 = mu_r_2_and_b[0]
                        mu_r_2 = mu_r_2_and_b[1]
                        t_mu_phi_2 = interp1d(b_phi_deg_2, mu_phi_deg_2)
                        t_mu_r_2 = interp1d(b_r_2, mu_r_2)
                # --------linear interpolation at constant freq-------------
                mu_phi = []
                mu_r = []
                b_t = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2]

                for y in range(len(b_t)):
                    mu_r.append(
                        t_mu_r_1(b_t[y]) + (t_mu_r_2(b_t[y]) - t_mu_r_1(b_t[y])) / (
                                t_2 - t_1) * (variable - t_1))
                    mu_phi.append(
                        t_mu_phi_1(b_t[y]) + (t_mu_phi_2(b_t[y]) - t_mu_phi_1(b_t[y])) / (
                                t_2 - t_1) * (variable - t_1))
                return mu_r, mu_phi

            # --------interpolated data at constant freq and nearby temp--------
            interpolate_temp_1 = getdata_measurements(T, f_l, temp_list_l[0], temp_list_l[1])
            interpolate_temp_2 = getdata_measurements(T, f_h, temp_list_h[0], temp_list_h[1])
            # print(interpolate_temp_1)
            # print(interpolate_temp_2)

            # ------linear interpolation at constant temp and nearby freq-----------------
            self.b_f = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2]
            f_mu_r_1 = interp1d(self.b_f, interpolate_temp_1[0])
            f_mu_phi_1 = interp1d(self.b_f, interpolate_temp_1[1])
            f_mu_r_2 = interp1d(self.b_f, interpolate_temp_2[0])
            f_mu_phi_2 = interp1d(self.b_f, interpolate_temp_2[1])
            mu_phi_f = []
            mu_r_f = []
            for b in range(len(self.b_f)):
                mu_r_f.append(
                    f_mu_r_1(self.b_f[b]) + (f_mu_r_2(self.b_f[b]) - f_mu_r_1(self.b_f[b])) / (
                            f_h - f_l) * (
                            f - f_l))
                mu_phi_f.append(
                    f_mu_phi_1(self.b_f[b]) + (f_mu_phi_2(self.b_f[b]) - f_mu_phi_1(self.b_f[b])) / (
                            f_h - f_l) * (
                            f - f_l))
            mu_real_from_polar = []
            mu_imag_from_polar = []
            for n in range(len(self.b_f)):
                cartesian = rect(mu_r_f[n], mu_phi_f[n])
                mu_real_from_polar.append(cartesian[0])
                mu_imag_from_polar.append(cartesian[1])
            self.mu_real = mu_real_from_polar
            self.mu_imag = mu_imag_from_polar
            print(self.mu_real)
            print(self.mu_imag)


        elif datasource == "manufacturer_datasheet":

            m_data = data[f"{material_name}"][f"{datasource}"]
            m_data_new = m_data["permeability_data"]
            for j in range(len(m_data_new)):
                freq_list.append(m_data_new[j]["frequency"])

            # print(freq_list)
            n = len(freq_list)  # len of array
            freq_list = list(remove(freq_list, n))
            # print(freq_list)

            result = find_nearest(freq_list, f)
            # print(result)

            f_l = result[0]
            f_h = result[1]

            # ------find nearby temperature------
            temp_list_l = []
            temp_list_h = []

            for i in range(len(m_data_new)):
                if m_data_new[i]["frequency"] == f_l:
                    temp_list_l.append(m_data_new[i]["temperature"])
            for i in range(len(m_data_new)):
                if m_data_new[i]["frequency"] == f_h:
                    temp_list_h.append(m_data_new[i]["temperature"])

            temp_list_l = find_nearest(temp_list_l, T)
            temp_list_h = find_nearest(temp_list_h, T)

            # print(temp_list_l)

            # print(temp_list_h)

            # -------get the data----------
            def getdata(variable, F, t_1, t_2):
                for k in range(len(m_data_new)):
                    if m_data_new[k]["frequency"] == F and m_data_new[k]["temperature"] == t_1:
                        self.b_1 = m_data_new[k]["b"]
                        mu_real_1 = m_data_new[k]["mu_real"]
                        mu_imag_1 = m_data_new[k]["mu_imag"]
                        t_mu_imag_1 = interp1d(self.b_1, mu_imag_1)
                        t_mu_real_1 = interp1d(self.b_1, mu_real_1)
                    if m_data_new[k]["frequency"] == F and \
                            m_data_new[k]["temperature"] == t_2:
                        self.b_2 = m_data_new[k]["b"]
                        mu_real_2 = m_data_new[k]["mu_real"]
                        mu_imag_2 = m_data_new[k]["mu_imag"]
                        t_mu_imag_2 = interp1d(self.b_2, mu_imag_2)
                        t_mu_real_2 = interp1d(self.b_2, mu_real_2)

                # --------linear interpolation at constant freq-------------
                mu_i = []
                mu_r = []
                # b_t = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

                for y in range(len(self.b_1)):
                    mu_r.append(
                        t_mu_real_1(self.b_1[y]) + (t_mu_real_2(self.b_1[y]) - t_mu_real_1(self.b_1[y])) / (
                                t_2 - t_1) * (
                                variable - t_1))
                    mu_i.append(
                        t_mu_imag_1(self.b_1[y]) + (t_mu_imag_2(self.b_1[y]) - t_mu_imag_1(self.b_1[y])) / (
                                t_2 - t_1) * (
                                variable - t_1))
                return mu_r, mu_i

            # --------interpolated data at constant freq and nearby temp--------
            interpolate_temp_1 = getdata(T, f_l, temp_list_l[0], temp_list_l[1])
            interpolate_temp_2 = getdata(T, f_h, temp_list_h[0], temp_list_h[1])
            # print(interpolate_temp_1)
            # print(interpolate_temp_2)

            # ------linear interpolation at constant temp and nearby freq-----------------
            # self.b_f = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            f_mu_real_1 = interp1d(self.b_1, interpolate_temp_1[0])
            f_mu_imag_1 = interp1d(self.b_1, interpolate_temp_1[1])
            f_mu_real_2 = interp1d(self.b_1, interpolate_temp_2[0])
            f_mu_imag_2 = interp1d(self.b_1, interpolate_temp_2[1])
            mu_i_f = []
            mu_r_f = []
            for b in range(len(self.b_1)):
                mu_r_f.append(
                    f_mu_real_1(self.b_1[b]) + (f_mu_real_2(self.b_1[b]) - f_mu_real_1(self.b_1[b])) / (f_h - f_l) * (
                            f - f_l))
                mu_i_f.append(
                    f_mu_imag_1(self.b_1[b]) + (f_mu_imag_2(self.b_1[b]) - f_mu_imag_1(self.b_1[b])) / (f_h - f_l) * (
                            f - f_l))
            self.mu_real = mu_r_f
            self.mu_imag = mu_i_f
            # print(self.mu_real)
            # print(self.mu_imag)

        else:
            raise Exception("'datasource' must be 'manufacturer_datasheet' or 'measurements'.")

        if pro:
            self.export_data(parent_directory=parent_directory, file_format="pro")
        mdb_print(f"Material properties of {material_name} are loaded at {T} °C and {f} Hz.")
        return self.b_1, self.mu_imag, self.mu_real

    def export_data(self, parent_directory: str = "", file_format: str = None):
        """
        Method is used to export data from the material database in a certain file format.
        :param file_format: export format
        :parent_directory:
        @param file_format:
        @param parent_directory:
        """
        if file_format == "pro":
            with open(os.path.join(parent_directory, "core_materials_temp.pro"), "w") as file:
                file.write(f'Include "Parameter.pro";\n')
                file.write(
                    f"Function{{\n  b = {str(self.b_1).replace('[', '{').replace(']', '}')} ;\n  mu_real = {str(self.mu_real).replace('[', '{').replace(']', '}')} ;"
                    f"\n  mu_imag = {str(self.mu_imag).replace('[', '{').replace(']', '}')} ;\n  "
                    f"mu_imag_couples = ListAlt[b(), mu_imag()] ;\n  "
                    f"mu_real_couples = ListAlt[b(), mu_real()] ;\n  "
                    f"f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{{List[mu_imag_couples]}};\n  "
                    f"f_mu_real_d[] = InterpolationLinear[Norm[$1]]{{List[mu_real_couples]}};\n  "
                    f"f_mu_imag[] = f_mu_imag_d[$1];\n  "
                    f"f_mu_real[] = f_mu_real_d[$1];\n }}  ")

        else:
            raise Exception("No valid file format is given!")

        mdb_print(f"Data is exported to {parent_directory} in a {file_format}-file.")

    def store_data(self, material_name, data_to_be_stored):
        """
        Method is used to store data from measurement/datasheet into the material database.
        :param material_name:
        :param data_to_be_stored:
        :return:
        """
        with open('material_data_base.json', 'w') as outfile:
            json.dump(data_to_be_stored, outfile, indent=4)
        mdb_print(f"Material properties of {material_name} are stored in the material database.")

    def plot_data(self, material_name: str = None, properties: str = None):
        """
        Method is used to plot certain material properties of materials.
        :param properties:
        :param material_name:
        :return:
        """
        if properties == "mu_real":
            plt.plot(self.b_f, self.mu_real)
            plt.ylabel(properties)
            plt.xlabel('B in T')
            plt.title("Real part of permeability")
            plt.show()
        elif properties == "mu_imag":
            plt.plot(self.b_f, self.mu_imag)
            plt.ylabel(properties)
            plt.xlabel('B in T')
            plt.title("Imaginary part of permeability")
            plt.show()

        mdb_print(f"Material properties {properties} of {material_name} are plotted.")

    # --------to get different material property from database file---------
    @staticmethod
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
        mdb_print(f'value=', value)
        return value

    # ----------to get steinmetz data from database file-----------------------
    @staticmethod
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
        else:
            raise Exception(
                "Error in selecting loss data. 'type' must be 'Steinmetz' or others (will be implemented in future).")
        # elif type == "Generalized_Steinmetz":
        #     coefficient = dict(s_data[f"{material_name}"]["generalized_steinmetz_data"])
        # print(coefficient)
        return coefficient


def drop_down_list(material_name: str, comparison_type: str, temperature: bool = False, flux: bool = False,
                   freq: bool = False):
    """
    This function return a list temp, frq nad flux to GUI from database to used as dropdown list
    @param freq: to get freq list
    @param material_name:
    @param temperature: to get temp list
    @param flux: to get flux list
    @param comparison_type: datasheet vs datasheet ="dvd", measurement vs measurement = "mvm", datasheet vs measurement = "dvm"
    @return:
    """
    global temp_list, flux_list, freq_list_new, temp_list_new, flux_list_new
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)

    # ------looking for temperatures and flux values in database----
    # ------ datasheet vs datasheet-----------
    if comparison_type == "dvd":
        curve_data_material = curve_data[f"{material_name}"]["manufacturer_datasheet"]
        temp_list = []
        for i in range(len(curve_data_material["b_h_curve"])):
            temp_list.append(curve_data_material["b_h_curve"][i]["temperature"])
        for i in range(len(curve_data_material["relative_core_loss_flux_density"])):
            temp_list.append(curve_data_material["relative_core_loss_flux_density"][i]["temperature"])
        for i in range(len(curve_data_material["relative_core_loss_frequency"])):
            temp_list.append(curve_data_material["relative_core_loss_frequency"][i]["temperature"])

        flux_list = []
        for i in range(len(curve_data_material["relative_core_loss_temperature"])):
            flux_list.append(curve_data_material["relative_core_loss_temperature"][i]["b"])
        for i in range(len(curve_data_material["relative_core_loss_frequency"])):
            flux_list.append(curve_data_material["relative_core_loss_frequency"][i]["b"])
        temp_list_new = list(remove(temp_list, len(temp_list)))
        flux_list_new = list(remove(flux_list, len(flux_list)))
        temp_list_new.sort()
        flux_list_new.sort()

    # ------- measurement vs measurement------
    if comparison_type == "mvm":
        curve_data_material = curve_data[f"{material_name}"]["measurements"]
        temp_list = []
        freq_list = []
        for j in range(len(curve_data_material)):
            if curve_data_material[j]["data_type"] == "complex_permeability_data":
                curve_data_material_new = curve_data_material[j]["permeability_data"]
                for i in range(len(curve_data_material_new)):
                    temp_list.append(curve_data_material_new[i]["temperature"])

                for i in range(len(curve_data_material_new)):
                    freq_list.append(curve_data_material_new[i]["frequency"])

        temp_list_new = list(remove(temp_list, len(temp_list)))
        freq_list_new = list(remove(freq_list, len(freq_list)))
        temp_list_new.sort()
        freq_list_new.sort()
    if temperature:
        return temp_list_new
    if flux:
        return flux_list_new
    if freq:
        return freq_list_new


def material_list_in_database(material_list: bool = True):
    """

    @param material_list: boolean to get material list from the database for GUI
    @return: materials present in database in form of list.
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        data = json.load(data)
    materials = []
    for i in data:
        materials.append(i)
    return materials


def compare_core_loss_flux_density_data(matplotlib_widget, material_list: list, temperature_list: list = None):
    """
    Method is used to compare material properties.
    :param material_list:[material1, material2, .....]
    :param temperature_list
    :return:
    """
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)
    # print(material_list)

    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"][
            "relative_core_loss_flux_density"]
        material = material_list[i]
        temperature = temperature_list[i]
        b = []
        frequency = []
        power_loss = []
        color = color_list[i]
        for j in range(len(curve_data_material)):
            if curve_data_material[j]["temperature"] == temperature:
                b.append(curve_data_material[j]["b"])
                frequency.append(curve_data_material[j]["frequency"])
                power_loss.append(curve_data_material[j]["power_loss"])
        for j in range(len(b)):
            line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                          (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]
            label = f"{material}", f"F={frequency[j]}Hz", f"T={temperature}°C"
            lines = matplotlib_widget.axis.plot(b[j], power_loss[j], label=label, color=color, linestyle=line_style[j])
            mplcursors.cursor(lines)
            # plt.plot(b[j], power_loss[j], label=label, color=color, linestyle=line_style[j])
            # plt.legend()
    matplotlib_widget.axis.set(xlabel="B in T", ylabel="Relative power loss in W/m\u00b3", yscale='log', xscale='log')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel("B in T")
    # plt.ylabel("Relative power loss in W/m\u00b3")
    # plt.grid()
    # plt.show()

    mdb_print(f"Material properties of {material_list} are compared.")


def compare_core_loss_temperature(matplotlib_widget, material_list: list, flux_list: list = None):
    """
        Method is used to compare material properties.
        :param material_list:[material1, material2, ....]
        :param flux_list
        :return:
        """
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)
    # print(material_list)

    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"][
            "relative_core_loss_temperature"]
        material = material_list[i]
        flux = flux_list[i]
        temperature = []
        frequency = []
        power_loss = []
        color = color_list[i]
        for j in range(len(curve_data_material)):
            if curve_data_material[j]["b"] == flux:
                temperature.append(curve_data_material[j]["temperature"])
                frequency.append(curve_data_material[j]["frequency"])
                power_loss.append(curve_data_material[j]["power_loss"])
        for j in range(len(temperature)):
            line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                          (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]
            label = f"{material}", f"F={frequency[j]}Hz", f"B= {flux}T"
            lines = matplotlib_widget.axis.plot(temperature[j], power_loss[j], label=label, color=color,
                                                linestyle=line_style[j])
            mplcursors.cursor(lines)
            # plt.plot(temperature[j], power_loss[j], label=label, color=color, linestyle=line_style[j])
            # plt.legend()
    matplotlib_widget.axis.set(xlabel="Temperature in °C", ylabel="Relative power loss in W/m\u00b3", yscale='log')
    # plt.yscale('log')
    # plt.xlabel("Temperature in °C")
    # plt.ylabel("Relative power loss in W/m\u00b3")
    # plt.grid()
    # plt.show()
    mdb_print(f"Material properties of {material_list} are compared.")


def compare_core_loss_frequency(matplotlib_widget, material_list: list, temperature_list: list = None,
                                flux_list: list = None):
    """
            Method is used to compare material properties.
            :param material_list:[material1, material2, ....]
            :param flux_list
            :param temperature_list
            :return:
            """
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)

    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"][
            "relative_core_loss_frequency"]
        material = material_list[i]
        temperature = temperature_list[i]
        flux = flux_list[i]
        frequency = []
        power_loss = []
        color = color_list[i]
        for m in range(len(curve_data_material)):
            if curve_data_material[m]["temperature"] == temperature and curve_data_material[m]["b"] == flux:
                frequency.append(curve_data_material[m]["frequency"])
                power_loss.append(curve_data_material[m]["power_loss"])

        for j in range(len(frequency)):
            line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                          (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]
            label = f"{material}", f"B= {flux}T", f"T= {temperature}°C"
            lines = matplotlib_widget.axis.plot(frequency[j], power_loss[j], color=color, label=label,
                                                linestyle=line_style[j])
            mplcursors.cursor(lines)
            # plt.plot(frequency[j], power_loss[j], color=color, label=label, linestyle=line_style[j])
            # plt.legend()
    matplotlib_widget.axis.set(xlabel="Frequency in Hz", ylabel="Relative power loss in W/m\u00b3", yscale='log',
                               xscale='log')
    # plt.grid()
    # plt.show()
    mdb_print(f"Material properties of {material_list} are compared.")


def compare_b_h_curve(matplotlib_widget, material_list: list, temperature_list: list = None):
    # -------B_H Curve-------
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as B_H:
        curve_data = json.load(B_H)

    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"][
            "b_h_curve"]
        b = []
        h = []
        color = color_list[i]
        material = material_list[i]
        temperature = temperature_list[i]
        for m in range(len(curve_data_material)):
            if curve_data_material[m]["temperature"] == temperature:
                b.append(curve_data_material[m]["b"])
                h.append(curve_data_material[m]["h"])
        for j in range(len(b)):
            line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                          (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]
            label = f"{material}", f"T= {temperature}°C"
            lines = matplotlib_widget.axis.plot(h[j], b[j], label=label, color=color, linestyle=line_style[j])
            mplcursors.cursor(lines)
            # plt.legend()
    matplotlib_widget.axis.set(xlabel="H in A/m", ylabel="B in T")
    # plt.ylabel('B')
    # plt.xlabel('H')
    # plt.title(f"B_H curve")
    # plt.grid()
    # plt.show()
    mdb_print(f"Material properties of {material_list} are compared.")


def compare_permeability_measurement_data(matplotlib_widget, material_list: list, frequency_list: list = None,
                                          temperature_list: list = None, plot_real_part: bool = False):
    """
        Method is used to compare material properties.
        :param material_list:[material1, material2, .....]
        @param plot_real_part: True plot real part of mu/ False plots imaginary part of mu
        @type temperature_list: object
        @param material_list:
        @param frequency_list:
        :return:
        """
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].grid()
    # axs[1].grid()
    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["measurements"]
        material = material_list[i]
        temperature = temperature_list[i]
        frequency = frequency_list[i]
        color = color_list[i]

        for j in range(len(curve_data_material)):
            if curve_data_material[j]["data_type"] == "complex_permeability_data":
                curve_data_material_new = curve_data_material[j]["permeability_data"]
                b = []
                freq = []
                mu_phi = []
                mu_r = []

                for k in range(len(curve_data_material_new)):
                    if curve_data_material_new[k]["frequency"] == frequency and curve_data_material_new[k][
                        "temperature"] == temperature:
                        b.append(curve_data_material_new[k]["b"])
                        freq.append(curve_data_material_new[k]["frequency"])
                        mu_phi.append(curve_data_material_new[k]["mu_phi_deg"])
                        mu_r.append(curve_data_material_new[k]["mu_r"])

                for k in range(len(b)):
                    if plot_real_part:
                        line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)),
                                      (0, (5, 10)),
                                      (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)),
                                      (0, (3, 10, 1, 10))]
                        label = f"{material}", f"T={temperature}°C"
                        # plt.plot(mu_r[k][0], mu_r[k][1], label=label, color=color, linestyle=line_style[k])
                        # plt.xlabel(r"B in T")
                        # plt.ylabel(r"$\mu_\mathrm{r}  /  \mu_0$")
                        # plt.legend()
                        lines = matplotlib_widget.axis.plot(mu_r[k][0], mu_r[k][1], label=label, color=color,
                                                            linestyle=line_style[k])
                        mplcursors.cursor(lines)
                        matplotlib_widget.axis.set(xlabel=r"B in T", ylabel=r"$\mu_\mathrm{r}  /  \mu_0$")

                    else:
                        line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)),
                                      (0, (5, 10)),
                                      (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)),
                                      (0, (3, 10, 1, 10))]
                        label = f"{material}", f"T={temperature}°C"
                        # plt.plot(b[k], mu_phi[k], label=label, color=color, linestyle=line_style[k])
                        # plt.xlabel(r"B in T")
                        # plt.ylabel(r"$\zeta_\mathrm{\mu}$")
                        # plt.legend()
                        lines = matplotlib_widget.axis.plot(b[k], mu_phi[k], label=label, color=color,
                                                            linestyle=line_style[k])
                        mplcursors.cursor(lines)
                        matplotlib_widget.axis.set(xlabel=r"B in T", ylabel=r"$\mu_\mathrm{r}  /  \mu_0$")

                    # mplcursors.cursor(lines_1)
                    # mplcursors.cursor(lines_2)

    # plt.show()
    mdb_print(f"Material properties of {material_list} are compared.")


def compare_core_loss_flux_datasheet_measurement(matplotlib_widget, material: str, temperature_list: list = None):
    """
    Method is used to compare material properties in datasheet and measurement.
    @param material_list:
    @param temperature_list:
    @return:
    """
    color_list = ['red', 'blue', 'green', 'yellow', 'orange']
    line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                  (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)

    curve_data_material_datasheet = curve_data[f"{material}"]["manufacturer_datasheet"][
        "relative_core_loss_flux_density"]
    curve_data_material_measurement = curve_data[f"{material}"]["measurements"]
    temperature_datasheet = temperature_list[0]
    temperature_measurement = temperature_list[1]
    b_d = []
    frequency_d = []
    power_loss_d = []
    b_m = []
    frequency_m = []
    power_loss_m = []

    for j in range(len(curve_data_material_datasheet)):
        if curve_data_material_datasheet[j]["temperature"] == temperature_datasheet:
            b_d.append(curve_data_material_datasheet[j]["b"])
            frequency_d.append(curve_data_material_datasheet[j]["frequency"])
            power_loss_d.append(curve_data_material_datasheet[j]["power_loss"])
    for j in range(len(b_d)):
        label = f"{material}", f"F={frequency_d[j]}Hz", f"T={temperature_datasheet}°C", f"Datasheet"
        lines = matplotlib_widget.axis.plot(b_d[j], power_loss_d[j], label=label, color=color_list[0], linestyle=line_style[0])
        mplcursors.cursor(lines)
        # plt.plot(b_d[j], power_loss_d[j], label=label, color=color_list[0], linestyle=line_style[0])
        # plt.legend()
    for j in range(len(curve_data_material_measurement)):
        if curve_data_material_measurement[j]["data_type"] == "complex_permeability_data":
            curve_data_material_measurement_new = curve_data_material_measurement[j]["core_loss_flux_density"]
            for j in range(len(curve_data_material_measurement_new)):
                if curve_data_material_measurement_new[j]["temperature"] == temperature_measurement:
                    b_m.append(curve_data_material_measurement_new[j]["b"])
                    frequency_m.append(curve_data_material_measurement_new[j]["frequency"])
                    power_loss_m.append(curve_data_material_measurement_new[j]["power_loss"])
            for j in range(len(b_m)):
                label = f"{material}", f"F={frequency_m[j]}Hz", f"T={temperature_measurement}°C", f"Measurements"
                lines = matplotlib_widget.axis.plot(b_m[j], power_loss_m[j], label=label, color=color_list[1], linestyle=line_style[1])
                mplcursors.cursor(lines)
                # plt.plot(b_m[j], power_loss_m[j], label=label, color=color_list[1], linestyle=line_style[1])
                # plt.legend()
    matplotlib_widget.axis.set(xlabel="B in T", ylabel="Relative power loss in W/m\u00b3", yscale='log', xscale='log')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel("B in T")
    # plt.ylabel("Relative power loss in W/m\u00b3")
    # plt.grid()
    # plt.show()
    mdb_print(f"Material properties of {material} are compared.")