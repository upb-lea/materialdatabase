"""Script to write the data of manufacturer datasheets plots into the database."""
from matplotlib import pyplot as plt
import numpy as np
from materialdatabase.enumerations import *
from materialdatabase.material_data_base_functions import *
from materialdatabase.paths import *
import os

"""
    Information regarding Digitization:
        - used program: WebPlotDigiter (made by Ankit Rohatgi)
        - format to save: Sort by: X
                          Order: Ascending  
                          Digits: 5 Fixed
                          Column Separator: ;
        - use units in graphs for digitization and refactor with variables down below
                        
    IMPORTANT Information regarding BH-curves:
        - BH-curves have to be sampled until 0 A/m
        - for the BH-curve two datasets are needed: 1. the lower curve
                                                    2. the upper curve
        - lower curve -> Sort by: X
                         Order: Ascending
        - upper curve -> Sort by: X
                         Order: Descending 

    Information regarding Units in the Database:
        
       - Temperature in Degree Celsius
       - Frequency in Hertz
       - Magnetic flux density in Tesla
       - Magnetic field strength in Ampere per Meter
       - Powerloss in Watt per cubic Meter
"""

WRITE = False
# MAGNETIC FIELD STRENGTH
Oersted_TO_Ampere_Per_Meter = False
h_field_factor = 1
# MAGNETIC FLUX DENSITY
MilliTesla_TO_Tesla = True
Gauss_TO_Tesla = False
b_field_factor = 1
# POWERLOSS
KiloWatt_Per_Cubic_Meter_TO_Watt_Per_Cubic_Meter = True
powerloss_factor = 1
# FREQUENCY
MegaHertz_TO_Hertz = False
KiloHertz_TO_Hertz = False
frequency_factor = 1

if Oersted_TO_Ampere_Per_Meter:
    h_field_factor = 1000/4/np.pi

if MilliTesla_TO_Tesla:
    b_field_factor = 1e-3
if Gauss_TO_Tesla:
    b_field_factor = 1e-4

if KiloWatt_Per_Cubic_Meter_TO_Watt_Per_Cubic_Meter:
    powerloss_factor = 1e3

if MegaHertz_TO_Hertz:
    frequency_factor = 1e-6
if KiloHertz_TO_Hertz:
    frequency_factor = 1e-3

material = str(Material._3F46.value)
manufacturer = str(Manufacturer.Ferroxcube.value)

# PATHS TO DATA | DATA FORMAT UNDER VARIABLE
# complex_permeability_frequency = []
complex_permeability_frequency = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "complex_permeability_real.csv"),
                                   os.path.join(datasheet_path, manufacturer, material + "_digitized", "complex_permeability_imag.csv"), 25)]
# [(path to .csv-file with real part, path to .csv-file with imagnary part, temperature), ...]

# initial_permeability_temperature = []
initial_permeability_temperature = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "init_permeability_temperature.csv"))]
# [(path to .csv-file), ...]

initial_permeability_frequency = []
# initial_permeability_frequency = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "init_permeability_frequency.csv"))]
# [(path to .csv-file), ...]

incremental_permeability_field_strength = []
# incremental_permeability_field_strength = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "permeability_over_h_dc.csv"), 10e3, 25)]
# [(path to .csv-file), ...]

# amplitude_permeability_flux_density = []
amplitude_permeability_flux_density = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "amplitude_permeability_25C_25kHz.csv"), 25e3,
                                        25),
                                       (os.path.join(datasheet_path, manufacturer, material + "_digitized", "amplitude_permeability_100C_25kHz.csv"), 25e3,
                                        100),
                                       (os.path.join(datasheet_path, manufacturer, material + "_digitized", "amplitude_permeability_100C_1000kHz.csv"), 1000e3,
                                        100),
                                       (os.path.join(datasheet_path, manufacturer, material + "_digitized", "amplitude_permeability_100C_3000kHz.csv"), 3000e3,
                                        100)]
# [(path to .csv-file, frequency, temperature), ...]

# bh_curves = []
bh_curves = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "bh_curve_25C_lower.csv"),
              os.path.join(datasheet_path, manufacturer, material + "_digitized", "bh_curve_25C_upper.csv"), 10e3, 25),
             (os.path.join(datasheet_path, manufacturer, material + "_digitized", "bh_curve_100C_lower.csv"),
              os.path.join(datasheet_path, manufacturer, material + "_digitized", "bh_curve_100C_upper.csv"), 10e3, 100)]
# [(path to .csv-file (lower_curve), path to .csv-file (upper_curve), frequency, temperature), ...]

# relative_core_loss_flux_density = []
relative_core_loss_flux_density = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_flux_density_100C_1000kHz.csv"),
                                    1000e3, 100),
                                   (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_flux_density_100C_2000kHz.csv"),
                                    2000e3, 100),
                                   (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_flux_density_100C_3000kHz.csv"),
                                    3000e3, 100)]
# [(path to .csv-file, frequency, temperature), ...]

# relative_core_loss_temperature = []
relative_core_loss_temperature = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_temperature_1000kHz_50mT.csv"), 1000e3, 0.05),
                                  (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_temperature_3000kHz_10mT.csv"), 3000e3, 0.01),
                                  (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_temperature_3000kHz_30mT.csv"), 3000e3, 0.03)]
# [(path to .csv-file, frequency, flux_density), ...]

relative_core_loss_frequency = []
# relative_core_loss_frequency = [(os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_25mT_20C.csv"), 20, 0.025),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_25mT_100C.csv"), 100, 0.025),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_50mT_20C.csv"), 20, 0.05),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_50mT_100C.csv"), 100, 0.05),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_100mT_20C.csv"), 20, 0.1),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_100mT_100C.csv"), 100, 0.1),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_200mT_20C.csv"), 20, 0.2),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_200mT_100C.csv"), 100, 0.2),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_300mT_20C.csv"), 20, 0.3),
#                                 (os.path.join(datasheet_path, manufacturer, material + "_digitized", "powerloss_over_frequency_300mT_100C.csv"), 100, 0.3)]
# [(path to .csv-file, temperature, flux_density), ...]

if WRITE:
    with open(relative_path_to_db, "r") as jsonFile:
        database = json.load(jsonFile)
else:
    database = {material: {"manufacturer_datasheet": {}}}

# COMPLEX PERMEABILITY VERSUS FREQUENCY --------------------------------------------------------------------------------------------------------------------
if not complex_permeability_frequency:
    pass
else:
    data_list = []

    for data in complex_permeability_frequency:
        df_real = read_in_digitized_datasheet_plot(data[0])
        df_imag = read_in_digitized_datasheet_plot(data[1])

        imag_part_interpolated = updates_x_ticks_for_graph(x_data=df_imag[0], y_data=df_imag[1], x_new=df_real[0], kind="linear")
        imag_part_interpolated = np.array([x if x >= 1 else 1 for x in imag_part_interpolated])  # set imag part to 1 if value less than 1 (AFTER INTERPOLATION)

        fig, ax = plt.subplots(1, 1)
        plt.loglog(np.array(df_real[0])*frequency_factor, df_real[1], label="original real")
        plt.loglog(np.array(df_imag[0])*frequency_factor, df_imag[1], label="original imag")
        plt.loglog(np.array(df_real[0])*frequency_factor, imag_part_interpolated, label="interpolate imag", linestyle="--")
        ax.set_xlabel(PlotLabels.frequency_Hz.value)
        ax.set_ylabel(PlotLabels.mu_ampl.value)
        plt.title("Complex-Permeability")
        plt.legend()
        plt.grid(True, which="both")
        plt.show()

        mu_r_complex = np.array(df_real[1]) + j*imag_part_interpolated
        mu_r_abs = abs(mu_r_complex)
        mu_phi_deg = np.angle(mu_r_complex, deg=True)

        data_dict = {"mu_r_abs": list(mu_r_abs),
                     "mu_phi_deg": list(mu_phi_deg),
                     "frequency": list(np.array(df_imag[0])*frequency_factor),
                     "temperature": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.complex_permeability] = data_list

# INITIAL PERMEABILITY VERSUS FREQUENCY ------------------------------------------------------------------------------------------------------------------
if not initial_permeability_frequency:
    pass
else:
    data_list = []

    for data in initial_permeability_frequency:
        df = read_in_digitized_datasheet_plot(data)

        fig, ax = plt.subplots(1, 1)
        plt.plot(df[0], df[1])
        ax.set_xlabel(PlotLabels.frequency_Hz.value)
        ax.set_ylabel(PlotLabels.mu_init.value)
        plt.title("Initial-Permeability")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"initial_permeability": list(df[1]),
                     "frequency": list(df[0])}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.initial_permeability_frequency] = data_list


# INITIAL PERMEABILITY VERSUS TEMPERATURE ------------------------------------------------------------------------------------------------------------------
if not initial_permeability_temperature:
    pass
else:
    data_list = []

    for data in initial_permeability_temperature:
        df = read_in_digitized_datasheet_plot(data)

        fig, ax = plt.subplots(1, 1)
        plt.plot(df[0], df[1])
        ax.set_xlabel(PlotLabels.temperature_in_C.value)
        ax.set_ylabel(PlotLabels.mu_init.value)
        plt.title("Initial-Permeability")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"initial_permeability": list(df[1]),
                     "temperature": list(df[0])}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.initial_permeability_temperature] = data_list


# INCREMENTAL PERMEABILITY VERSUS FIELD STRENGTH----------------------------------------------------------------------------------------------------------------
if not incremental_permeability_field_strength:
    pass
else:
    data_list = []

    for data in incremental_permeability_field_strength:
        df = read_in_digitized_datasheet_plot(data[0])

        fig, ax = plt.subplots(1, 1)
        plt.loglog(np.array(df[0])*h_field_factor, df[1])
        ax.set_xlabel(PlotLabels.h_field.value)
        ax.set_ylabel(r"rel. permeability incremental $\mu_\mathrm{r}}$")
        plt.title("Incremental-Permeability")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"incremental_permeability": list(df[1]),
                     "field_strength": list(np.array(df[0])*h_field_factor),
                     "frequency": data[1],
                     "temperature": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.incremental_permeability_field_strength] = data_list

# AMPLITUDE PERMEABILITY VERSUS FLUX DENSITY ---------------------------------------------------------------------------------------------------------------
if not amplitude_permeability_flux_density:
    pass
else:
    data_list = []

    for data in amplitude_permeability_flux_density:
        df = read_in_digitized_datasheet_plot(data[0])

        fig, ax = plt.subplots(1, 1)
        plt.plot(np.array(df[0])*b_field_factor, df[1])
        ax.set_xlabel(PlotLabels.b_field.value)
        ax.set_ylabel(PlotLabels.mu_ampl.value)
        plt.title("Amplitude-Permeability @ " + str(data[2]) + "°C")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"mu_r_abs": list(df[1]),
                     "flux_density": list(np.array(df[0])*b_field_factor),
                     "frequency": data[1],
                     "temperature": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.amplitude_permeability_flux_density] = data_list

# HYSTERESIS-CURVE -----------------------------------------------------------------------------------------------------------------------------------------
if not bh_curves:
    pass
else:
    data_list = []

    for data in bh_curves:
        df_lower = read_in_digitized_datasheet_plot(data[0])
        df_upper = read_in_digitized_datasheet_plot(data[1])

        complete_bh_curve = [df_lower[0] + df_upper[0], df_lower[1] + df_upper[1]]

        fig, ax = plt.subplots(1, 1)
        plt.plot(np.array(complete_bh_curve[0])*h_field_factor, np.array(complete_bh_curve[1])*b_field_factor)
        ax.set_xlabel(PlotLabels.h_field.value)
        ax.set_ylabel(PlotLabels.b_field.value)
        plt.title("BH-Curve @ " + str(data[3]) + "°C")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"flux_density": list(np.array(complete_bh_curve[1])*b_field_factor),
                     "field_strength": list(np.array(complete_bh_curve[0])*h_field_factor),
                     "frequency": data[2],
                     "temperature": data[3]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.b_h_curve] = data_list

# RELATIVE CORE LOSS VERSUS FLUX DENSITY -------------------------------------------------------------------------------------------------------------------
if not relative_core_loss_flux_density:
    pass
else:
    data_list = []
    # [(path to .csv-file, frequency, temperature), ...]
    for data in relative_core_loss_flux_density:
        df = read_in_digitized_datasheet_plot(data[0])

        fig, ax = plt.subplots(1, 1)
        plt.loglog(np.array(df[0])*b_field_factor, np.array(df[1])*powerloss_factor)
        ax.set_xlabel(PlotLabels.b_field.value)
        ax.set_ylabel(PlotLabels.powerloss_density_W.value)
        plt.title("Powerloss @ " + str(data[2]) + "°C & " + str(data[1]) + "Hz")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"flux_density": list(np.array(df[0])*b_field_factor),
                     "power_loss": list(np.array(df[1])*powerloss_factor),
                     "frequency": data[1],
                     "temperature": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.relative_core_loss_flux_density] = data_list

# RELATIVE CORE LOSS VERSUS TEMPERATURE --------------------------------------------------------------------------------------------------------------------
if not relative_core_loss_temperature:
    pass
else:
    data_list = []
    # [(path to .csv-file, frequency, flux_density), ...]
    for data in relative_core_loss_temperature:
        df = read_in_digitized_datasheet_plot(data[0])

        fig, ax = plt.subplots(1, 1)
        plt.semilogy(df[0], np.array(df[1])*powerloss_factor)
        ax.set_xlabel(PlotLabels.temperature_in_C.value)
        ax.set_ylabel(PlotLabels.powerloss_density_W.value)
        plt.title("Powerloss @ " + str(data[2]) + "T & " + str(data[1]) + "Hz")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"temperature": list(df[0]),
                     "power_loss": list(np.array(df[1])*powerloss_factor),
                     "frequency": data[1],
                     "flux_density": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.relative_core_loss_temperature] = data_list

# RELATIVE CORE LOSS VERSUS FREQUENCY ----------------------------------------------------------------------------------------------------------------------
if not relative_core_loss_frequency:
    pass
else:
    data_list = []
    # [(path to .csv-file, frequency, flux_density), ...]
    for data in relative_core_loss_frequency:
        df = read_in_digitized_datasheet_plot(data[0])
        fig, ax = plt.subplots(1, 1)
        plt.loglog(np.array(df[0])*frequency_factor, np.array(df[1])*powerloss_factor)
        ax.set_xlabel(PlotLabels.frequency_Hz.value)
        ax.set_ylabel(PlotLabels.powerloss_density_W.value)
        plt.title("Powerloss @ " + str(data[1]) + "°C & " + str(data[2]) + "T")
        plt.grid(True, which="both")
        plt.show()

        data_dict = {"frequency": list(np.array(df[0])*frequency_factor),
                     "power_loss": list(np.array(df[1])*powerloss_factor),
                     "temperature": data[1],
                     "flux_density": data[2]}
        data_list.append(data_dict)

    database[material][MaterialDataSource.ManufacturerDatasheet][DatasheetPlotName.relative_core_loss_frequency] = data_list

if WRITE:
    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(database, jsonFile, indent=2)
