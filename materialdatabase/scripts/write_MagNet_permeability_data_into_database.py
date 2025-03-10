"""Script to write MagNet-Data into the database and on local drive."""
import mat73
import numpy as np
from materialdatabase.enumerations import *
from materialdatabase.material_data_base_functions import *
from materialdatabase import paths as pt
import pandas as pd
import os
# from scipy import constants
import materialdatabase as mdb

material_db = mdb.MaterialDatabase()

"""
  PREPERATIONS:
    1. DOWNLOAD MagNet-Data(Single Cycle) FROM DROPBOX 
       (https://www.dropbox.com/scl/fo/jx6itx4nna5d4ki4dbxue/ANIJVaD_TFm8UzP5C-358n0?rlkey=248tctc9u2b39erd6sv9j99pa&e=1&dl=0
       OR https://www.princeton.edu/~minjie/magnet.html) AND PUT .mat-FILE IN DESIRED FOLDER. NAME OF THE FOLDER SHOULD BE THE NAME OF THE MATERIAL
    2. CREATE IN DESIRED FOLDER 3 SUBFOLDER WITH THE NAMES "sine", "trapezoid" and "triangle" 

  INSTRUCTIONS TO PUT DATA INTO DATABASE:
    0. CHOOSE SIGNAL-SHAPE, TO LOAD OR PROCESS THE DATA AND TO WRITE THE DATA INTO THE DATABASE
    1. SET MINIMUM NUMBER OF MEASUREMENT POINTS (START WITH LOW VALUE)
    2. INPUT MATERIAL NAME AS ENUMERATION (Material.XXX.value)
    3. INPUT MANUFACTURER NAME AS ENUMERATION (Manufacturer.XXX)
    4. INPUT BASIC DATASHEET PARAMETERS FOR MATERIAL
    5. INPUT PATH TO FOLDER CONTAINING MagNet-DATA
    6. CHANGE ENUMERATION IN VARIABLE datadict TO DESIRED MATERIAl (MagNetFileNames._XXX.value)
    7. RUN SCRIPT
"""

# SIGNAL-SHAPE
SINE = True  # SET TRUE TO RUN SINE DATA
TRIANGLE = False  # SET TRUE TO RUN TRIANGULAR DATA  # TODO NOT IMPLEMENTED
TRAPEZOID = False  # SET TRUE TO RUN TRAPEZOIDAL DATA  # TODO NOT IMPLEMENTED
# PROCESS OF DATA
PROCESS_DATA = True  # SET TRUE TO PROCESS DATA IF NOT DATA IS LOADED FROM GIVEN PATH
# WRITE DATA INTO DATABASE
WRITE = False  # SET TRUE TO WRITE DATA INTO DATABASE

min_number_of_measurements = 7  # 1.

material = Material._3F4.value   # 2.
manufacturer = Manufacturer.Ferroxcube  # 3.
initial_permeability = 900  # 4.
resistivity = 10  # 4.
max_flux_density = 0.41  # 4.
volumetric_mass_density = 4700  # 4.

path = os.path.join(pt.my_MagNet_data_path, material)  # 5.
data_dict = mat73.loadmat(os.path.join(path, MagNetFileNames._3F4.value))  # 6.

cross_section = data_dict["Data"]["Effective_Area"]
l_mag = data_dict["Data"]["Effective_Length"]
volume = data_dict["Data"]["Effective_Volume"]
primary_windings = data_dict["Data"]["Primary_Turns"]
secondary_windings = data_dict["Data"]["Secondary_Turns"]
date = data_dict["Data"]["Date_processing"]
probe = data_dict["Data"]["Shape"]

if PROCESS_DATA:
    # filter for duty-cycle is NaN
    sine_bool = [True if np.isnan(inner_loop) else False for inner_loop in data_dict["Data"]["DutyP_command"]]
    # filter for dutyP + dutyN == 1
    triangle_bool = [True if x + y == 1 else False for x, y in zip(data_dict["Data"]["DutyP_command"], data_dict["Data"]["DutyN_command"])]
    # filter for dutyP + dutyN != 1 and not NaN
    trapezoidal_bool = [True if (x + y != 1) and (not np.isnan(x + y)) else False
                        for x, y in zip(data_dict["Data"]["DutyP_command"], data_dict["Data"]["DutyN_command"])]
    print("Filter bools created!")

# SINE ---------------------------------------------------------------------------------------------------------------------------------------------------------
if SINE:
    if PROCESS_DATA:
        voltage = data_dict["Data"]["Voltage"][sine_bool]
        current = data_dict["Data"]["Current"][sine_bool]
        H_DC_Bias = data_dict["Data"]["Hdc_command"][sine_bool]
        temperature = data_dict["Data"]["Temperature_command"][sine_bool]
        frequency = data_dict["Data"]["Frequency_command"][sine_bool]

        mag_flux_density = [calc_magnetic_flux_density_based_on_voltage_array_and_frequency(voltage=v, frequency=f, secondary_winding=secondary_windings,
                                                                                            cross_section=cross_section) for v, f in zip(voltage, frequency)]
        print("Sine: Magnetic flux density calculated!")

        mag_field_strength_offset = [calc_magnetic_field_strength_based_on_current_array(current=i, primary_winding=primary_windings, l_mag=l_mag)
                                     for i in current]
        print("Sine: Magnetic field strength calculated!")

        mag_field_strength_without_offset = [h - offset for h, offset in zip(mag_field_strength_offset, H_DC_Bias)]
        # mag_field_strength_without_offset = [remove_mean_of_signal(h) for h in mag_field_strength_offset]
        permeability_amplitude = [calc_mu_r_from_b_and_h_array(b=b, h=h) for b, h in zip(mag_flux_density, mag_field_strength_without_offset)]

        # B_DC = np.array(H_DC_Bias) * mu
        # B_DC = np.array(H_DC_Bias) * constants.mu_0 * np.array(permeability_amplitude)
        # mag_flux_density_offset = [b + offset for b, offset in zip(mag_flux_density, B_DC)]

        powerloss = [abs(f * np.trapz(u * i, np.linspace(0, 1/f, len(list(u)))) / volume) for u, i, f in zip(voltage, current, frequency)]
        # powerloss_BH = [get_bh_integral_shoelace(b=b, h=h, f=f) for b, h, f in zip(mag_flux_density, mag_field_strength_without_offset, frequency)]

        permeability_angle = [mu_phi_deg__from_mu_r_and_p_hyst(frequency=f, b_peak=(max(b) - min(b))/2, mu_r=mu, p_hyst=p)
                              for f, b, mu, p in zip(frequency, mag_flux_density, permeability_amplitude, powerloss)]

        np.savetxt(os.path.join(path, "sine/Voltage[V].csv"), voltage, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Current[A].csv"), current, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Frequency[Hz].csv"), frequency, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Temperature[C].csv"), temperature, delimiter=",")
        np.savetxt(os.path.join(path, "sine/H_waveform[Am-1].csv"), mag_field_strength_without_offset, delimiter=",")
        np.savetxt(os.path.join(path, "sine/H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
        np.savetxt(os.path.join(path, "sine/B_waveform[T].csv"), mag_flux_density, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Volumetric_losses[Wm-3].csv"), powerloss, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Permeability_amplitude[_].csv"), permeability_amplitude, delimiter=",")
        np.savetxt(os.path.join(path, "sine/Permeability_angle[°].csv"), permeability_angle, delimiter=",")

        dict_sine = {"temperature": temperature,
                     "mag_flux_density": [(max(b) - min(b))/2 for b in mag_flux_density],
                     "frequency": (np.array(frequency)/1000).astype(int)*1000,  # round up to kHz
                     "powerloss": powerloss,
                     "H_DC_Bias": H_DC_Bias,
                     "permeability_amplitude": permeability_amplitude,
                     "permeability_angle": permeability_angle}

        df_sine = pd.DataFrame.from_dict(dict_sine)
        df_sine.to_csv(os.path.join(path, "sine/data_sine.csv"), index=False)
        print("\n Sine data processed and saved!")

    else:
        df_sine = pd.read_csv(os.path.join(path, "sine/data_sine.csv"), encoding='latin1')

    unique_frequency = sorted(set(df_sine["frequency"]))
    # unique_H_DC_offset = sorted(set(df_sine["H_DC_Bias"]))
    unique_H_DC_offset = [0]  # only the data for an H_DC of 0 A/m
    unique_temperature = sorted(set(df_sine["temperature"]))

    # Init the database entry
    if WRITE:
        create_empty_material(material_name=material, manufacturer=manufacturer, initial_permeability=initial_permeability, resistivity=resistivity,
                              max_flux_density=max_flux_density, volumetric_mass_density=volumetric_mass_density)
        create_permeability_measurement_in_database(material, measurement_setup="MagNet", company="Princeton", date=date, test_setup_name="MagNet",
                                                    toroid_dimensions=probe, measurement_method="tba", equipment_names="tba")

    filter_string = "temperature == @temperature and H_DC_Bias == @H_DC and frequency == @frequency"

    for temperature in unique_temperature:
        for H_DC in unique_H_DC_offset:
            for frequency in unique_frequency:
                print('\033[1m' + "Number of measurement points: ", df_sine.query(filter_string).shape[0])
                if df_sine.query(filter_string).shape[0] >= min_number_of_measurements:
                    b_ref = np.array(df_sine.query(filter_string).sort_values('mag_flux_density')["mag_flux_density"])
                    mu_r = np.array(df_sine.query(filter_string).sort_values('mag_flux_density')["permeability_amplitude"])
                    mu_phi_deg = np.array(df_sine.query(filter_string).sort_values('mag_flux_density')["permeability_angle"])

                    print("Temperature: ", temperature)
                    print("H_DC: ", H_DC)
                    print("Frequency: ", frequency)

                    b_ref, mu_r, mu_phi_deg = sort_data(b_ref, mu_r, mu_phi_deg)
                    b_ref, mu_r, mu_phi_deg = interpolate_a_b_c(b_ref, mu_r, mu_phi_deg)
                    b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref, mu_r, mu_phi_deg, smooth_data=True, crop_data=True, plot_data=True,
                                                                        f=frequency, b_min=0.005, b_max=0.4)

                    if WRITE:
                        write_permeability_data_into_database(current_shape="sine", frequency=frequency, temperature=temperature, H_DC_offset=H_DC, b_ref=b_ref,
                                                              mu_r_abs=mu_r, mu_phi_deg=mu_phi_deg, material_name=material, measurement_setup="MagNet")


# TRIANGLE -----------------------------------------------------------------------------------------------------------------------------------------------------
if TRIANGLE:
    if PROCESS_DATA:
        voltage = data_dict["Data"]["Voltage"][triangle_bool]
        current = data_dict["Data"]["Current"][triangle_bool]
        H_DC_Bias = data_dict["Data"]["Hdc_command"][triangle_bool]
        temperature = data_dict["Data"]["Temperature_command"][triangle_bool]
        frequency = data_dict["Data"]["Frequency_command"][triangle_bool]
        duty_cycle = data_dict["Data"]["DutyP_command"][triangle_bool]

        mag_flux_density = [calc_magnetic_flux_density_based_on_voltage_array_and_frequency(voltage=v, frequency=f, secondary_winding=secondary_windings,
                                                                                            cross_section=cross_section) for v, f in zip(voltage, frequency)]
        mag_flux_density = [b - np.mean(b) for b in mag_flux_density]
        print("Triangle: Magnetic flux density calculated!")

        mag_field_strength_offset = [calc_magnetic_field_strength_based_on_current_array(current=i, primary_winding=primary_windings, l_mag=l_mag)
                                     for i in current]
        print("Triangle: Magnetic field strength calculated!")

        mag_field_strength_without_offset = [remove_mean_of_signal(h) for h in mag_field_strength_offset]
        permeability_amplitude = [calc_mu_r_from_b_and_h_array(b=b, h=h) for b, h in zip(mag_flux_density, mag_field_strength_without_offset)]

        # B_DC = np.array(H_DC_Bias) * mu
        # B_DC = np.array(H_DC_Bias) * constants.mu_0 * np.array(permeability_amplitude)
        # mag_flux_density_offset = [b + offset for b, offset in zip(mag_flux_density, B_DC)]

        powerloss = [get_bh_integral_shoelace(b=b, h=h, f=f) for b, h, f in zip(mag_flux_density, mag_field_strength_offset, frequency)]

        permeability_angle = [mu_phi_deg__from_mu_r_and_p_hyst(frequency=f, b_peak=abs(max(b, key=abs)), mu_r=mu, p_hyst=p)
                              for f, b, mu, p in zip(frequency, mag_flux_density, permeability_amplitude, powerloss)]

        np.savetxt(os.path.join(path, "triangle/Voltage[V].csv"), voltage, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Current[A].csv"), current, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Temperature[C].csv"), temperature, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Frequency[Hz].csv"), frequency, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Duty_cycle[_].csv"), duty_cycle, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/H_waveform[Am-1].csv"), mag_field_strength_offset, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/B_waveform[T].csv"), mag_flux_density, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Volumetric_losses[Wm-3].csv"), powerloss, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Permeability_amplitude[_].csv"), permeability_amplitude, delimiter=",")
        np.savetxt(os.path.join(path, "triangle/Permeability_angle[°].csv"), permeability_angle, delimiter=",")

        dict_triangle = {"temperature": temperature,
                         "mag_flux_density": [abs(max(x, key=abs)) for x in mag_flux_density],
                         "frequency": (np.array(frequency)/1000).astype(int)*1000,  # round up to kHz
                         "powerloss": powerloss,
                         "H_DC_Bias": H_DC_Bias,
                         "duty_cycle": duty_cycle,
                         "permeability_amplitude": permeability_amplitude,
                         "permeability_angle": permeability_angle}

        df_triangle = pd.DataFrame.from_dict(dict_triangle)
        df_triangle.to_csv(os.path.join(path, "triangle/data_triangle.csv"), index=False)
        print("Triangle data processed and saved!")

    else:
        df_sine = pd.read_csv(os.path.join(path, "triangle/data_triangle.csv"), encoding='latin1')

    unique_frequency = sorted(set(df_sine["frequency"]))
    unique_H_DC_offset = sorted(set(df_sine["H_DC_Bias"]))
    unique_temperature = sorted(set(df_sine["temperature"]))
    min_number_of_measurements = 8

    # Init the database entry
    if WRITE:
        create_permeability_measurement_in_database(material, measurement_setup="MagNet", company="Princeton", date=date, test_setup_name="MagNet",
                                                    toroid_dimensions=probe, measurement_method="tba", equipment_names="tba")

# TRAPEZOID ----------------------------------------------------------------------------------------------------------------------------------------------------
if TRAPEZOID:
    voltage = data_dict["Data"]["Voltage"][trapezoidal_bool]
    current = data_dict["Data"]["Current"][trapezoidal_bool]
    H_DC_Bias = data_dict["Data"]["Hdc_command"][trapezoidal_bool]
    temperature = data_dict["Data"]["Temperature_command"][trapezoidal_bool]
    frequency = data_dict["Data"]["Frequency_command"][trapezoidal_bool]

    np.savetxt(os.path.join(path, "trapezoid/Voltage[V].csv"), voltage, delimiter=",")
    np.savetxt(os.path.join(path, "trapezoid/Current[A].csv"), current, delimiter=",")
    np.savetxt(os.path.join(path, "trapezoid/H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
    np.savetxt(os.path.join(path, "trapezoid/Frequency[Hz].csv"), frequency, delimiter=",")
    np.savetxt(os.path.join(path, "trapezoid/Temperature[C].csv"), temperature, delimiter=",")

    print("Trapezoid data processed and saved!")
