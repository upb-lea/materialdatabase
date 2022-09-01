import os
import femmt as fmt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from .material_data_base_functions import *


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
        self.b_f = None
        self.mu_real = None
        self.mu_imag = None

    def get_permeability_data(self, T: float, f: int, material_name: str, datasource: str):
        """
        Method is used to read permeability data from the material database.
        :param T:
        :param f:
        :param material_name:
        :param datasource: measurement or datasheet
        :return:
        """
        self.temp = T
        self.freq = f
        self.mat = material_name

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'data/material_data_base.json')
        with open(file_path, 'r') as database:
            data = json.load(database)
        m_data = data[f"{material_name}"][f"{datasource}"]
        freq_list = []
        # print(len(m_data["permeability_data"]))
        for i in range(len(m_data)):
            if m_data[i]["data_type"] == "complex_permeability_data":
                m_data_new = m_data[i]["data"]
                for j in range(len(m_data[i]["data"])):
                    freq_list.append(m_data[i]["data"][j]["frequency"])
        print(freq_list)
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

        for i in range(len(m_data_new)):
            if m_data_new[i]["frequency"] == f_l:
                temp_list_l.append(m_data_new[i]["temperature"])
        for i in range(len(m_data_new)):
            if m_data_new[i]["frequency"] == f_h:
                temp_list_h.append(m_data_new[i]["temperature"])

        temp_list_l = find_nearest(temp_list_l, T)
        temp_list_h = find_nearest(temp_list_h, T)
        print(temp_list_l)

        # print(temp_list_h)

        # -------get the data----------
        def getdata(variable, F, t_1, t_2):
            for k in range(len(m_data_new)):
                if m_data_new[k]["frequency"] == F and m_data_new[k]["temperature"] == t_1:
                    b_1 = m_data_new[k]["b"]
                    mu_real_1 = m_data_new[k]["mu_real"]
                    mu_imag_1 = m_data_new[k]["mu_imag"]
                    t_mu_imag_1 = interp1d(b_1, mu_imag_1)
                    t_mu_real_1 = interp1d(b_1, mu_real_1)
                if m_data_new[k]["frequency"] == F and \
                        m_data_new[k]["temperature"] == t_2:
                    b_2 = m_data_new[k]["b"]
                    mu_real_2 = m_data_new[k]["mu_real"]
                    mu_imag_2 = m_data_new[k]["mu_imag"]
                    t_mu_imag_2 = interp1d(b_2, mu_imag_2)
                    t_mu_real_2 = interp1d(b_2, mu_real_2)

            # --------linear interpolation at constant freq-------------
            mu_i = []
            mu_r = []
            b_t = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

            for j in range(len(b_t)):
                mu_r.append(
                    t_mu_real_1(b_t[j]) + (t_mu_real_2(b_t[j]) - t_mu_real_1(b_t[j])) / (t_2 - t_1) * (variable - t_1))
                mu_i.append(
                    t_mu_imag_1(b_t[j]) + (t_mu_imag_2(b_t[j]) - t_mu_imag_1(b_t[j])) / (t_2 - t_1) * (variable - t_1))
            return mu_r, mu_i

        # --------interpolated data at constant freq and nearby temp--------
        interpolate_temp_1 = getdata(T, f_l, temp_list_l[0], temp_list_l[1])
        interpolate_temp_2 = getdata(T, f_h, temp_list_h[0], temp_list_h[1])
        # print(interpolate_temp_1)
        # print(interpolate_temp_2)

        # ------linear interpolation at constant temp and nearby freq-----------------
        self.b_f = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        f_mu_real_1 = interp1d(self.b_f, interpolate_temp_1[0])
        f_mu_imag_1 = interp1d(self.b_f, interpolate_temp_1[1])
        f_mu_real_2 = interp1d(self.b_f, interpolate_temp_2[0])
        f_mu_imag_2 = interp1d(self.b_f, interpolate_temp_2[1])
        mu_i_f = []
        mu_r_f = []
        for b in range(len(self.b_f)):
            mu_r_f.append(
                f_mu_real_1(self.b_f[b]) + (f_mu_real_2(self.b_f[b]) - f_mu_real_1(self.b_f[b])) / (f_h - f_l) * (
                        f - f_l))
            mu_i_f.append(
                f_mu_imag_1(self.b_f[b]) + (f_mu_imag_2(self.b_f[b]) - f_mu_imag_1(self.b_f[b])) / (f_h - f_l) * (
                        f - f_l))
        self.mu_real = mu_r_f
        self.mu_imag = mu_i_f
        print(self.mu_real)
        print(self.mu_imag)
        with open("core_materials_temp.pro", "w") as file:
            file.write(f'Include "Parameter.pro";\n')
            file.write(
                f"Function{{\n  b = {str(self.b_f).replace('[', '{').replace(']', '}')} ;\n  mu_real = {str(self.mu_real).replace('[', '{').replace(']', '}')} ;"
                f"\n  mu_imag = {str(self.mu_imag).replace('[', '{').replace(']', '}')} ;\n  "
                f"mu_imag_couples = ListAlt[b(), mu_imag()] ;\n  "
                f"mu_real_couples = ListAlt[b(), mu_real()] ;\n  "
                f"f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{{List[mu_imag_couples]}};\n  "
                f"f_mu_real_d[] = InterpolationLinear[Norm[$1]]{{List[mu_real_couples]}};\n  "
                f"f_mu_imag[] = f_mu_imag_d[$1];\n  "
                f"f_mu_real[] = f_mu_real_d[$1];\n }}  ")
        print(f"Material properties of {material_name} are loaded at {T} °C and {f} Hz.")
        pass

    def export_data(self, file_format: str = None):
        """
        Method is used to export data from the material database in a certain file format.
        :param data_to_export:
        :param format:
        :return:
        """
        if file_format == "pro":
            with open("core_materials_temp.pro", "w") as file:
                file.write(f'Include "Parameter.pro";\n')
                file.write(
                    f"Function{{\n  b = {str(self.b_f).replace('[', '{').replace(']', '}')} ;\n  mu_real = {str(self.mu_real).replace('[', '{').replace(']', '}')} ;"
                    f"\n  mu_imag = {str(self.mu_imag).replace('[', '{').replace(']', '}')} ;\n  "
                    f"mu_imag_couples = ListAlt[b(), mu_imag()] ;\n  "
                    f"mu_real_couples = ListAlt[b(), mu_real()] ;\n  "
                    f"f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{{List[mu_imag_couples]}};\n  "
                    f"f_mu_real_d[] = InterpolationLinear[Norm[$1]]{{List[mu_real_couples]}};\n  "
                    f"f_mu_imag[] = f_mu_imag_d[$1];\n  "
                    f"f_mu_real[] = f_mu_real_d[$1];\n }}  ")

        print(f"Data is exported in a {file_format}-file.")
        pass

    def store_data(self, material_name, data_to_be_stored):
        """
        Method is used to store data from measurement/datasheet into the material database.
        :param material_name:
        :param data_to_be_stored:
        :return:
        """
        with open('material_data_base.json', 'w') as outfile:
            json.dump(data_to_be_stored, outfile, indent=4)
        print(f"Material properties of {material_name} are stored in the material database.")
        pass

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

        # -------B_H Curve-------
        if properties == "b_h_curve":
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, 'data/material_data_base.json')
            with open(file_path, 'r') as B_H:
                data = json.load(B_H)
            temp_list = []
            for i in range(len(data[f"{material_name}"][f"{properties}"])):
                temp_list.append(data[f"{material_name}"][f"{properties}"][i]["temperature"])
            print(temp_list)
            temp_list = find_nearest(temp_list, self.temp)

            def get_b_h(t_1, t_2):
                for k in range(len(data[f"{material_name}"][f"{properties}"])):
                    if data[f"{material_name}"][f"{properties}"][k]["temperature"] == t_1:
                        b_1 = data[f"{material_name}"][f"{properties}"][k]["b"]
                        h_1 = data[f"{material_name}"][f"{properties}"][k]["h"]
                        b_h_1 = interp1d(h_1, b_1)
                        plt.plot(h_1, b_1)
                        plt.ylabel('B')
                        plt.xlabel('H')
                        plt.show()
                        plt.show()
                    if data[f"{material_name}"][f"{properties}"][k]["temperature"] == t_2:
                        b_2 = data[f"{material_name}"][f"{properties}"][k]["b"]
                        h_2 = data[f"{material_name}"][f"{properties}"][k]["h"]
                        plt.plot(h_2, b_2)
                        plt.ylabel('B')
                        plt.xlabel('H')
                        plt.show()
                        b_h_2 = interp1d(b_2, h_2)
                b = []
                h = h_2
                for j in range(len(h_2)):
                    b.append(b_h_1(h_2) + (b_h_2(h_2) - b_h_1(h_2)) / (t_2 - t_1) * (self.temp - t_1))
                return b, h

            curve = get_b_h(temp_list[0], temp_list[1])

            plt.plot(curve[1], curve[2])
            plt.ylabel('B')
            plt.xlabel('H')
            plt.title(f"B_H curve at {self.temp}°C")
            plt.show()

        print(f"Material properties {properties} of {material_name} are plotted.")
        pass

    # ------load Steinmetz data--------------
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
        # elif type == "Generalized_Steinmetz":
        #     coefficient = dict(s_data[f"{material_name}"]["generalized_steinmetz_data"])
        # print(coefficient)
        return coefficient

    # ----------load initial permeability---------
    @staticmethod
    def get_initial_permeability(material_name: str):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'data/material_data_base.json')
        with open(file_path, 'r') as data:
            ip_data = json.load(data)
        mu_rel = ip_data[f"{material_name}"]["manufacturer_datasheet"]["initial_permeability"]
        # print(mu_rel)
        return mu_rel

    # ----load resistivity-----------
    @staticmethod
    def get_resistivity(material_name: str):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'data/material_data_base.json')
        with open(file_path, 'r') as data:
            r_data = json.load(data)
        resistivity = r_data[f"{material_name}"]["manufacturer_datasheet"]["resistivity"]
        # print(resistivity)
        return resistivity


def compare_core_loss_flux_density_data(material_list: list, temperature: float):
    """
    Method is used to compare material properties.
    :param material_list:[material1, material2, .....]
    :param temperature
    :return:
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)
    print(material_list)
    b = []
    frequency = []
    power_loss = []
    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"][
            "relative_core_loss_flux_density"]
        for m in range(len(curve_data_material)):
            if curve_data_material[m]["temperature"] == temperature:
                b.append(curve_data_material[m]["b"])
                frequency.append(curve_data_material[m]["frequency"])
                power_loss.append(curve_data_material[m]["power_loss"])

    for i in range(len(material_list)):
        label = f"{material_list[i]}", f"frequency={frequency[i]}", f"temperature={temperature}"
        plt.plot(b[i], power_loss[i], label=label)
        plt.legend()

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("B in T")
    plt.ylabel("Relative power loss in W/m\u00b3")
    plt.show()
    print(f"Material properties of {material_list} are compared.")


def compare_core_loss_temperature(material_list: list, flux: float = None):
    """
        Method is used to compare material properties.
        :param material_list:[material1, material2, ....]
        :param flux
        :return:
        """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data/material_data_base.json')
    with open(file_path, 'r') as data:
        curve_data = json.load(data)
    print(material_list)
    temperature = []
    frequency = []
    power_loss = []
    for i in range(len(material_list)):
        curve_data_material = curve_data[f"{material_list[i]}"]["manufacturer_datasheet"]["relative_core_loss_temperature"]
        for m in range(len(curve_data_material)):
            if curve_data_material[m]["b"] == flux:
                temperature.append(curve_data_material[m]["temperature"])
                frequency.append(curve_data_material[m]["frequency"])
                power_loss.append(curve_data_material[m]["power_loss"])

    for i in range(len(material_list)):
        label = f"{material_list[i]}", f"Flux_density= {flux}"
        plt.plot(temperature[i], power_loss[i], label=label)
        plt.legend()
    plt.yscale('log')
    plt.xlabel("Temperature in °C")
    plt.ylabel("Relative power loss in W/m\u00b3")
    plt.show()
    print(f"Material properties of {material_list} are compared.")
