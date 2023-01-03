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

        self.database_file_directory = 'data/'
        self.database_file_name = 'material_data_base.json'
        self.data_folder_path = os.path.join(os.path.dirname(__file__), self.database_file_directory)
        self.data_file_path = os.path.join(self.data_folder_path, self.database_file_name)

        self.data = self.load_database()

        set_silent_status(is_silent)
        mdb_print("The material database is now initialized")

    def permeability_data_to_pro_file(self, T: float, f: float, material_name: str, datatype: str,
                                      datasource: str, measurement_setup: str, parent_directory: str = ""):
        """
        Method is used to read permeability data from the material database.
        :param T: temperature
        :param f: Frequency
        :param material_name: "N95","N87"....
        :param datasource: "measurements" or "manufacturer_datasheet"
        :param datatype: "complex_permeability", "complex_permittivity" or "Steinmetz"
        :param measurement_setup: name of measuerement setup
        :param parent_directory: location of solver file
        """

        check_input_permeability_data(datasource, material_name, T, f)

        if datasource == "measurements":
            permeability_data = self.data[f"{material_name}"][f"measurements"][f"{datatype}"][f"{measurement_setup}"]["permeability_data"]
            # print(f"{permeability_data = }")
            # print(f"{len(permeability_data[1]['b']), len(permeability_data[0]['mu_r']) = }")

            # create_permeability_neighbourhood
            nbh = create_permeability_neighbourhood_measurement(T, f, permeability_data)
            # print(f"{nbh = }")
            # print(f"{len(nbh['T_low_f_low']['b']), len(nbh['T_low_f_low']['mu_r']) = }")

            b_ref, mu_r = interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f,
                                                                                        nbh["T_low_f_low"]["T"], nbh["T_high_f_low"]["T"],
                                                                                        nbh["T_low_f_low"]["f"], nbh["T_low_f_high"]["f"],
                                                                                        nbh["T_low_f_low"]["b"], nbh["T_low_f_low"]["mu_r"],
                                                                                        nbh["T_high_f_low"]["b"], nbh["T_high_f_low"]["mu_r"],
                                                                                        nbh["T_low_f_high"]["b"], nbh["T_low_f_high"]["mu_r"],
                                                                                        nbh["T_high_f_high"]["b"], nbh["T_high_f_high"]["mu_r"])

            b_ref, mu_phi_deg = interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f,
                                                                                              nbh["T_low_f_low"]["T"], nbh["T_high_f_low"]["T"],
                                                                                              nbh["T_low_f_low"]["f"], nbh["T_low_f_high"]["f"],
                                                                                              nbh["T_low_f_low"]["b"], nbh["T_low_f_low"]["mu_phi_deg"],
                                                                                              nbh["T_high_f_low"]["b"], nbh["T_high_f_low"]["mu_phi_deg"],
                                                                                              nbh["T_low_f_high"]["b"], nbh["T_low_f_high"]["mu_phi_deg"],
                                                                                              nbh["T_high_f_high"]["b"], nbh["T_high_f_high"]["mu_phi_deg"])

            # print(f"{b_ref, mu_r, mu_phi_deg = }")

            # Convert to cartesian
            mu_real_from_polar, mu_imag_from_polar = [], []
            for n in range(len(b_ref)):
                cartesian = rect(mu_r[n], mu_phi_deg[n])
                mu_real_from_polar.append(cartesian[0])
                mu_imag_from_polar.append(cartesian[1])
            mu_real = mu_real_from_polar
            mu_imag = mu_imag_from_polar

        elif datasource == "manufacturer_datasheet":
            permeability_data = self.data[f"{material_name}"][f"{datasource}"]["permeability_data"]
            # print(f"{permeability_data = }")

            # create_permeability_neighbourhood
            nbh = create_permeability_neighbourhood_datasheet(T, f, permeability_data)
            # print(f"{nbh = }")

            b_ref, mu_real = interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f,
                                                                                           nbh["T_low_f_low"]["T"], nbh["T_high_f_low"]["T"],
                                                                                           nbh["T_low_f_low"]["f"], nbh["T_low_f_high"]["f"],
                                                                                           nbh["T_low_f_low"]["b"], nbh["T_low_f_low"]["mu_real"],
                                                                                           nbh["T_high_f_low"]["b"], nbh["T_high_f_low"]["mu_real"],
                                                                                           nbh["T_low_f_high"]["b"], nbh["T_low_f_high"]["mu_real"],
                                                                                           nbh["T_high_f_high"]["b"], nbh["T_high_f_high"]["mu_real"])

            b_ref, mu_imag = interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f,
                                                                                           nbh["T_low_f_low"]["T"], nbh["T_high_f_low"]["T"],
                                                                                           nbh["T_low_f_low"]["f"], nbh["T_low_f_high"]["f"],
                                                                                           nbh["T_low_f_low"]["b"], nbh["T_low_f_low"]["mu_imag"],
                                                                                           nbh["T_high_f_low"]["b"], nbh["T_high_f_low"]["mu_imag"],
                                                                                           nbh["T_low_f_high"]["b"], nbh["T_low_f_high"]["mu_imag"],
                                                                                           nbh["T_high_f_high"]["b"], nbh["T_high_f_high"]["mu_imag"])


            print(f"{b_ref, mu_real, mu_imag = }")

        # Write the .pro-file
        export_data(parent_directory=parent_directory, file_format="pro", b_ref=list(b_ref), mu_real=list(mu_real), mu_imag=list(mu_imag))

        mdb_print(f"Material properties of {material_name} are loaded at {T} °C and {f} Hz.")

        return b_ref, mu_imag, mu_real

    # --------to get different material property from database file---------
    def get_material_property(self, material_name: str, property: str):
        """
            :param material_name: str: N95,N87.....
            :param property: str:  initial_permeability, resistivity, max_flux_density, weight_density
        """
        value = self.data[f"{material_name}"]["manufacturer_datasheet"][f"{property}"]
        mdb_print(f'value=', value)
        return value

    # ----------to get steinmetz data from database file-----------------------
    def get_steinmetz_data(self, material_name: str, type: str, datasource: str):
        """
        :param material_name:
        :param datasource: measurement or datasheet
        :param type: steinmetz or generalized steinmetz
        """
        s_data = self.data[f"{material_name}"][f"{datasource}"]
        if type == "Steinmetz":
            for i in range(len(s_data)):
                if s_data[i]["data_type"] == "steinmetz_data":
                    coefficient = dict(s_data[i]["data"])
        else:
            raise Exception(
                "Error in selecting loss data. 'type' must be 'Steinmetz' or others (will be implemented in future).")
        # elif type == "Generalized_Steinmetz":
        #     coefficient = dict(s_data[f"{material_name}"]["generalized_steinmetz_data"])
        # print(coefficient)
        return coefficient

    def load_database(self):
        with open(self.data_file_path, 'r') as database:
            return json.load(database)

    def drop_down_list(self, material_name: str, comparison_type: str, temperature: bool = False, flux: bool = False,
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

        # ------looking for temperatures and flux values in database----
        # ------ datasheet vs datasheet-----------
        if comparison_type == "dvd":
            curve_data_material = self.data[f"{material_name}"]["manufacturer_datasheet"]
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
            curve_data_material = self.data[f"{material_name}"]["measurements"]
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

    def material_list_in_database(self):
        """
        @return: materials present in database in form of list.
        """
        materials = []
        for i in self.data:
            materials.append(i)
        return materials

    def compare_core_loss_flux_density_data(self, matplotlib_widget, material_list: list, temperature_list: list = None):
        """
        Method is used to compare material properties.
        :param material_list:[material1, material2, .....]
        :param temperature_list
        :return:
        """
        color_list = ['red', 'blue', 'green', 'yellow', 'orange']
        # print(material_list)

        for i in range(len(material_list)):
            curve_data_material = self.data[f"{material_list[i]}"]["manufacturer_datasheet"][
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

    def compare_core_loss_temperature(self, matplotlib_widget, material_list: list, flux_list: list = None):
        """
            Method is used to compare material properties.
            :param material_list:[material1, material2, ....]
            :param flux_list
            :return:
            """
        color_list = ['red', 'blue', 'green', 'yellow', 'orange']
        # print(material_list)

        for i in range(len(material_list)):
            curve_data_material = self.data[f"{material_list[i]}"]["manufacturer_datasheet"][
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

    def compare_core_loss_frequency(self, matplotlib_widget, material_list: list, temperature_list: list = None,
                                    flux_list: list = None):
        """
                Method is used to compare material properties.
                :param material_list:[material1, material2, ....]
                :param flux_list
                :param temperature_list
                :return:
                """
        color_list = ['red', 'blue', 'green', 'yellow', 'orange']

        for i in range(len(material_list)):
            curve_data_material = self.data[f"{material_list[i]}"]["manufacturer_datasheet"][
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

    def compare_b_h_curve(self, matplotlib_widget, material_list: list, temperature_list: list = None):
        # -------B_H Curve-------
        color_list = ['red', 'blue', 'green', 'yellow', 'orange']

        for i in range(len(material_list)):
            curve_data_material = self.data[f"{material_list[i]}"]["manufacturer_datasheet"][
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

    def compare_permeability_measurement_data(self, matplotlib_widget, material_list: list, frequency_list: list = None,
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

        # fig, axs = plt.subplots(1, 2)
        # axs[0].grid()
        # axs[1].grid()
        for i in range(len(material_list)):
            curve_data_material = self.data[f"{material_list[i]}"]["measurements"]
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

    def compare_core_loss_flux_datasheet_measurement(self, matplotlib_widget, material: str, temperature_list: list = None):
        """
        Method is used to compare material properties in datasheet and measurement.
        @param material_list:
        @param temperature_list:
        @return:
        """
        color_list = ['red', 'blue', 'green', 'yellow', 'orange']
        line_style = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
                      (0, ()), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 5)), (0, (1, 10)), (0, (3, 10, 1, 10))]

        curve_data_material_datasheet = self.data[f"{material}"]["manufacturer_datasheet"]["relative_core_loss_flux_density"]
        curve_data_material_measurement = self.data[f"{material}"]["measurements"]
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

    # Permittivity Data
    def load_permittivity_measurement(self, material_name: str, datasource: str = "measurements",
                                      datatype: str = "complex_permittivity", measurement_setup: str = None):
        """

        :param material_name:
        :param datasource:
        :param datatype:
        :param measurement_setup:
        :return:
        """
        # Load all available permittivity data from datasource
        print(f"{material_name = }"
              f"{datasource = }"
              f"{datatype = }"
              f"{measurement_setup =}")
        return self.data[material_name][datasource][datatype][measurement_setup]["measurement_data"]

    def get_permittivity(self, T: float, f: float, material_name: str,
                         datasource: str = "measurements", datatype: str = "complex_permittivity", measurement_setup: str = None,
                         interpolation_type: str = "linear"):
        """
        Returns the complex permittivity for a certain operation point defined by temperature T and frequency f.
        :param measurement_setup:
        :param datatype:
        :param interpolation_type:
        :param datasource:
        :param T: float
        :param f: float
        :param material_name: str
        :return: complex
        """
        # Load the chosen permittivity data from the database
        list_of_permittivity_dicts = self.load_permittivity_measurement(material_name, datasource, datatype, measurement_setup)

        # Find the data, that is closest to the given operation point (T, f)
        neighbourhood = create_permittivity_neighbourhood(T=T, f=f, list_of_permittivity_dicts=list_of_permittivity_dicts)

        # Interpolate/Extrapolate the permittivity according to the given operation point
        if interpolation_type == "linear":
            epsilon_r, epsilon_phi_deg = interpolate_neighbours_linear(T=T, f=f, neighbours=neighbourhood)
        else:
            raise NotImplementedError

        return epsilon_r, epsilon_phi_deg
