"""Script to write MagNet-Data into the database and on local drive."""

# python packages
import logging
from pathlib import Path
# 3rd party libraries
import mat73  # Load MATLAB 7.3 .mat files into python (needs to be installed separately)
from scipy.constants import mu_0
import pandas as pd
import numpy as np
# own libraries
import materialdatabase as mdb
from materialdatabase.meta.data_enums import Material, MagNetFileNames

logger = logging.getLogger(__name__)

def integrate(x_data: np.ndarray | list, y_data: np.ndarray | list) -> np.ndarray:
    """
    Integrate the function y_data = f(x_data).

    :param x_data: x-axis
    :type x_data: ndarray or list
    :param y_data: y-axis
    :type y_data: ndarray or list
    :return: defined integral of y_data
    """
    data = [np.trapezoid(y_data[0:index], x_data[0:index]) for index, value in enumerate(x_data)]
    return np.array(data)


# SIGNAL-SHAPE
SINE = True  # Set true to run sine data
TRIANGLE = False  # Set true to run triangular data  # TODO NOT IMPLEMENTED
TRAPEZOID = False  # Set true to run trapezoidal data  # TODO NOT IMPLEMENTED
# PROCESS OF DATA
PROCESS_DATA = False  # Set true to process data if not data is loaded

material = Material.T37.value

# Get paths from config.toml and join them
path2_material = Path(mdb.get_user_paths().external_material_data).joinpath("MagNet").joinpath(material)
path2mdb_data = Path(mdb.get_user_paths().material_data).joinpath("complex_permeability").joinpath("MagNet")
# Load .mat file of material
MagNet_data_dict = mat73.loadmat(path2_material.joinpath(MagNetFileNames._T37.value))

cross_section = MagNet_data_dict["Data"]["Effective_Area"]
l_mag = MagNet_data_dict["Data"]["Effective_Length"]
volume = MagNet_data_dict["Data"]["Effective_Volume"]
primary_winding = MagNet_data_dict["Data"]["Primary_Turns"]
secondary_winding = MagNet_data_dict["Data"]["Secondary_Turns"]
probe = MagNet_data_dict["Data"]["Shape"]

if PROCESS_DATA:
    # Filter for duty-cycle is NaN
    sine_bool = [True if np.isnan(inner_loop) else False for inner_loop in MagNet_data_dict["Data"]["DutyP_command"]]
    # Filter for dutyP + dutyN == 1
    triangle_bool = [True if x + y == 1 else False for x, y in zip(MagNet_data_dict["Data"]["DutyP_command"], MagNet_data_dict["Data"]["DutyN_command"],
                                                                   strict=True)]
    # Filter for dutyP + dutyN != 1 and not NaN
    trapezoidal_bool = [True if (x + y != 1) and (not np.isnan(x + y)) else False
                        for x, y in zip(MagNet_data_dict["Data"]["DutyP_command"], MagNet_data_dict["Data"]["DutyN_command"], strict=True)]
    logger.info("Filter bools created!")

# SINE ---------------------------------------------------------------------------------------------------------------------------------------------------------
if SINE:
    if PROCESS_DATA:
        voltage = MagNet_data_dict["Data"]["Voltage"][sine_bool]
        current = MagNet_data_dict["Data"]["Current"][sine_bool]
        H_DC_Bias = MagNet_data_dict["Data"]["Hdc_command"][sine_bool]
        temperature = MagNet_data_dict["Data"]["Temperature_command"][sine_bool]
        frequency = MagNet_data_dict["Data"]["Frequency_command"][sine_bool]

        mag_flux_density = [integrate(np.linspace(0, 1/f, v.shape[0]), v)/secondary_winding/cross_section for v, f in zip(voltage, frequency, strict=True)]
        logger.info("Sine: Magnetic flux density calculated!")

        mag_field_strength_offset = [i * primary_winding / l_mag for i in current]
        logger.info("Sine: Magnetic field strength calculated!")

        mag_field_strength_without_offset = [h - offset for h, offset in zip(mag_field_strength_offset, H_DC_Bias, strict=True)]

        permeability_amplitude = [((max(b)-min(b))/2) / ((max(h)-min(h))/2) / mu_0 for b, h in zip(mag_flux_density, mag_field_strength_without_offset,
                                                                                                   strict=True)]

        power_loss = [abs(f * np.trapezoid(u*i, np.linspace(0, 1/f, len(list(u)))) / volume) for u, i, f in zip(voltage, current, frequency, strict=True)]

        permeability_angle = [np.rad2deg(np.arcsin(p * mu * mu_0 / (np.pi * f * ((max(b)-min(b))/2)**2)))
                              for f, b, mu, p in zip(frequency, mag_flux_density, permeability_amplitude, power_loss, strict=True)]

        np.savetxt(path2_material.joinpath("sine").joinpath("Voltage[V].csv"), voltage, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Current[A].csv"), current, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Frequency[Hz].csv"), frequency, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Temperature[C].csv"), temperature, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("H_waveform[Am-1].csv"), mag_field_strength_without_offset, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("B_waveform[T].csv"), mag_flux_density, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Volumetric_losses[Wm-3].csv"), power_loss, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Permeability_amplitude[_].csv"), permeability_amplitude, delimiter=",")
        np.savetxt(path2_material.joinpath("sine").joinpath("Permeability_angle[°].csv"), permeability_angle, delimiter=",")

        dict_sine = {"temperature": temperature,
                     "mag_flux_density": [(max(b) - min(b))/2 for b in mag_flux_density],
                     "frequency": np.round(frequency, -3),  # Round up to kHz
                     "power_loss": power_loss,
                     "H_DC_Bias": H_DC_Bias,
                     "permeability_amplitude": permeability_amplitude,
                     "permeability_angle": permeability_angle}

        df_sine = pd.DataFrame.from_dict(dict_sine)
        df_sine.to_csv(path2_material.joinpath("sine").joinpath("data_sine.csv"), index=False)
        logger.info("\n Sine data processed and saved!")

    else:
        df_sine = pd.read_csv(path2_material.joinpath("sine").joinpath("data_sine.csv"), encoding='latin1')
        logger.info("\n Sine data loaded!")

    unique_H_DC_offset = sorted(set(df_sine["H_DC_Bias"]))

    for H_DC in unique_H_DC_offset:
        mu_r = np.array(df_sine.query("H_DC_Bias == @H_DC")["permeability_amplitude"])
        mu_phi_deg = np.array(df_sine.query("H_DC_Bias == @H_DC")["permeability_angle"])
        mu_real = mu_r * np.cos(np.deg2rad(mu_phi_deg))
        mu_imag = mu_r * np.sin(np.deg2rad(mu_phi_deg))
        f = np.array(df_sine.query("H_DC_Bias == @H_DC")["frequency"])
        T = np.array(df_sine.query("H_DC_Bias == @H_DC")["temperature"])
        b = np.array(df_sine.query("H_DC_Bias == @H_DC")["mag_flux_density"])

        data_dict = {"f": f, "T": T, "b": b, "mu_real": mu_real, "mu_imag": mu_imag}

        if H_DC == 0:
            pd.DataFrame.from_dict(data_dict).to_csv(path2mdb_data.joinpath(material + ".csv"), index=False)
        else:
            pd.DataFrame.from_dict(data_dict).to_csv(path2mdb_data.joinpath(material + "_" + str(H_DC) + "Am-1.csv"), index=False)

# TRIANGLE -----------------------------------------------------------------------------------------------------------------------------------------------------
if TRIANGLE and False:
    if PROCESS_DATA:
        voltage = MagNet_data_dict["Data"]["Voltage"][triangle_bool]
        current = MagNet_data_dict["Data"]["Current"][triangle_bool]
        H_DC_Bias = MagNet_data_dict["Data"]["Hdc_command"][triangle_bool]
        temperature = MagNet_data_dict["Data"]["Temperature_command"][triangle_bool]
        frequency = MagNet_data_dict["Data"]["Frequency_command"][triangle_bool]
        duty_cycle = MagNet_data_dict["Data"]["DutyP_command"][triangle_bool]

        mag_flux_density = [integrate(np.linspace(0, 1/f, v.shape[0]), v)/secondary_winding/cross_section for v, f in zip(voltage, frequency, strict=True)]
        logger.info("Triangle: Magnetic flux density calculated!")

        mag_field_strength_offset = [i * primary_winding / l_mag for i in current]
        logger.info("Triangle: Magnetic field strength calculated!")

        mag_field_strength_without_offset = [h - offset for h, offset in zip(mag_field_strength_offset, H_DC_Bias, strict=True)]

        permeability_amplitude = [((max(b)-min(b))/2) / ((max(h)-min(h))/2) / mu_0 for b, h in zip(mag_flux_density, mag_field_strength_without_offset,
                                                                                                   strict=True)]

        power_loss = [abs(f * np.trapezoid(u*i, np.linspace(0, 1/f, len(list(u)))) / volume) for u, i, f in zip(voltage, current, frequency, strict=True)]

        permeability_angle = [np.rad2deg(np.arcsin(p * mu * mu_0 / (np.pi * f * ((max(b)-min(b))/2)**2)))
                              for f, b, mu, p in zip(frequency, mag_flux_density, permeability_amplitude, power_loss, strict=True)]

        np.savetxt(path2_material.joinpath("triangle").joinpath("Voltage[V].csv"), voltage, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Current[A].csv"), current, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Frequency[Hz].csv"), frequency, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Temperature[C].csv"), temperature, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("H_waveform[Am-1].csv"), mag_field_strength_without_offset, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("B_waveform[T].csv"), mag_flux_density, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Volumetric_losses[Wm-3].csv"), power_loss, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Permeability_amplitude[_].csv"), permeability_amplitude, delimiter=",")
        np.savetxt(path2_material.joinpath("triangle").joinpath("Permeability_angle[°].csv"), permeability_angle, delimiter=",")

        dict_triangle = {"temperature": temperature,
                         "mag_flux_density": [abs(max(x, key=abs)) for x in mag_flux_density],
                         "frequency": np.round(frequency, -3),  # Round up to kHz
                         "power_loss": power_loss,
                         "H_DC_Bias": H_DC_Bias,
                         "duty_cycle": duty_cycle,
                         "permeability_amplitude": permeability_amplitude,
                         "permeability_angle": permeability_angle}

        df_triangle = pd.DataFrame.from_dict(dict_triangle)
        df_triangle.to_csv(path2_material.joinpath("triangle").joinpath("data_triangle.csv"), index=False)
        logger.info("\n Triangle data processed and saved!")

    else:
        df_triangle = pd.read_csv(path2_material.joinpath("triangle").joinpath("data_triangle.csv"), encoding='latin1')
        logger.info("\n Triangle data loaded!")

# TRAPEZOID ----------------------------------------------------------------------------------------------------------------------------------------------------
if TRAPEZOID and False:
    voltage = MagNet_data_dict["Data"]["Voltage"][trapezoidal_bool]
    current = MagNet_data_dict["Data"]["Current"][trapezoidal_bool]
    H_DC_Bias = MagNet_data_dict["Data"]["Hdc_command"][trapezoidal_bool]
    temperature = MagNet_data_dict["Data"]["Temperature_command"][trapezoidal_bool]
    frequency = MagNet_data_dict["Data"]["Frequency_command"][trapezoidal_bool]

    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Voltage[V].csv"), voltage, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Current[A].csv"), current, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Frequency[Hz].csv"), frequency, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Temperature[C].csv"), temperature, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("H_waveform[Am-1].csv"), mag_field_strength_without_offset, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("H_DC_Bias[Am-1].csv"), H_DC_Bias, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("B_waveform[T].csv"), mag_flux_density, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Volumetric_losses[Wm-3].csv"), power_loss, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Permeability_amplitude[_].csv"), permeability_amplitude, delimiter=",")
    np.savetxt(path2_material.joinpath("trapezoid").joinpath("Permeability_angle[°].csv"), permeability_angle, delimiter=",")

    logger.info("Trapezoid data processed and saved!")
