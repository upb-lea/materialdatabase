"""Example script to demonstrate processing and fitting complex permittivity data using the ComplexPermittivity class from the materialdatabase package."""
import logging
import numpy as np
from matplotlib import pyplot as plt
import materialdatabase as mdb
from materialdatabase.processing.utils.math import mean_relative_absolute_error

# Configure logging to show info level messages
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def plot_parity_amplitude(eps_r_meas: np.ndarray, eps_r_fit: np.ndarray) -> None:
    """
    Create a parity plot comparing measured and fitted absolute permittivity values.

    The plot shows measured values on the x-axis and fitted values on the y-axis
    along with an ideal fit reference line.
    :param eps_r_meas: measured complex permittivity values
    :param eps_r_fit: fitted complex permittivity values
    """
    print(
        f"MRE (amplitude): {np.round(100 * mean_relative_absolute_error(np.abs(eps_r_meas), np.abs(eps_r_fit)), decimals=2)} %")
    plt.figure(figsize=(8, 6))
    plt.scatter(abs(eps_r_meas), np.abs(eps_r_fit), alpha=0.7, label="Fitted vs Measured")
    plt.plot([abs(eps_r_meas).min(), abs(eps_r_meas).max()],
             [abs(eps_r_meas).min(), abs(eps_r_meas).max()],
             'r--', label="Ideal Fit (y = x)")
    plt.xlabel("Measured absolute permittivity")
    plt.ylabel("Fitted absolute permittivity")
    plt.title("Permittivity Magnitude Measured vs Fitted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_parity_angle(delta_meas: np.ndarray, delta_fit: np.ndarray) -> None:
    """
    Create a parity plot comparing measured and fitted loss angles.

    The plot shows measured values on the x-axis and fitted values on the y-axis
    along with an ideal fit reference line.
    :param delta_meas: of measured loss angles in degrees
    :param delta_fit: of fitted loss angles in degrees
    """
    print(
        f"MRE (angle): {np.round(100 * mean_relative_absolute_error(delta_meas, delta_fit), decimals=2)} %")
    plt.figure(figsize=(8, 6))
    plt.scatter(delta_meas, delta_fit, alpha=0.7, label="Fitted vs Measured")
    plt.plot([delta_meas.min(), delta_meas.max()],
             [delta_meas.min(), delta_meas.max()],
             'r--', label="Ideal Fit (y = x)")
    plt.xlabel("Measured Loss Angle (deg)")
    plt.ylabel("Fitted Loss Angle (deg)")
    plt.title("Loss Angle Measured vs Fitted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_over_frequency_amplitude(T: np.ndarray, f: np.ndarray, eps_r_meas: np.ndarray, eps_r_fit: np.ndarray) -> None:
    """
    Plot the measured and fitted absolute permittivity as a function of frequency.

    (for each unique temperature level found in the data)
    :param T: temperature values for each measurement
    :param f: frequency values for each measurement
    :param eps_r_meas: measured complex permittivity values
    :param eps_r_fit: fitted complex permittivity values
    """
    unique_temps = np.unique(T)
    n_temps = len(unique_temps)
    fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)
    if n_temps == 1:
        axes = [axes]
    for ax, temp in zip(axes, unique_temps, strict=False):
        mask = T == temp
        f_vals = f[mask]
        measured_vals = abs(eps_r_meas)[mask]
        fitted_vals = np.abs(eps_r_fit)[mask]
        ax.plot(f_vals, measured_vals, 'o', label="Measured")
        ax.plot(f_vals, fitted_vals, 'x', label="Fitted")
        ax.set_title(f"Temperature = {temp} degC")
        ax.set_ylabel("Absolute permittivity")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()


def plot_over_frequency_angle(T: np.ndarray, f: np.ndarray, delta_meas: np.ndarray, delta_fit: np.ndarray) -> None:
    """
    Plot the measured and fitted loss angle as a function of frequency.

    (for each unique temperature level found in the data)
    :param T: temperature values for each measurement
    :param f: frequency values for each measurement
    :param delta_meas: measured loss angles in degrees
    :param delta_fit: fitted loss angles in degrees
    """
    unique_temps = np.unique(T)
    n_temps = len(unique_temps)
    fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)
    if n_temps == 1:
        axes = [axes]
    for ax, temp in zip(axes, unique_temps, strict=False):
        mask = T == temp
        f_vals = f[mask]
        measured_vals = delta_meas[mask]
        fitted_vals = delta_fit[mask]
        ax.plot(f_vals, measured_vals, 'o', label="Measured")
        ax.plot(f_vals, fitted_vals, 'x', label="Fitted")
        ax.set_title(f"Temperature = {temp} degC")
        ax.set_ylabel("Loss Angle (deg)")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()


def plot_over_frequency(T: np.ndarray, f: np.ndarray, eps_r_meas: np.ndarray, eps_r_fit: np.ndarray, delta_meas: np.ndarray, delta_fit: np.ndarray) -> None:
    """
    Wrap both amplitude and loss angle plots over frequency for each temperature level.

    :param T: temperature values for each measurement
    :param f: frequency values for each measurement
    :param eps_r_meas: measured complex permittivity values
    :param eps_r_fit: fitted complex permittivity values
    :param delta_meas: measured loss angles in degrees
    :param delta_fit: fitted loss angles in degrees
    """
    plot_over_frequency_amplitude(T, f, eps_r_meas, eps_r_fit)
    plot_over_frequency_angle(T, f, delta_meas, delta_fit)


def fit_complex_permittivity_example(is_plot: bool = True) -> None:
    """Fit the permittivity.

    :param is_plot: True to show visual outputs
    :type is_plot: bool
    """
    # Initialize the material database
    mdb_data = mdb.Data()
    # Load permittivity data for a specific material and setup
    permittivity = mdb_data.get_complex_permittivity(
        material=mdb.Material.N95,
        data_source=mdb.DataSource.LEA_MTB,
        probe_codes=["LE2"]
    )
    # permittivity = mdb_data.get_complex_permittivity(
    #     material=mdb.Material.N49,
    #     data_source=mdb.DataSource.LEA_MTB,
    #     probe_codes=["U3G"]
    # )
    # permittivity = mdb_data.get_complex_permittivity(
    #     material=mdb.Material._3F46,
    #     data_source=mdb.DataSource.LEA_MTB,
    #     probe_codes=["L95"]
    # )

    print("Exemplary complex permittivity data:")
    print(permittivity.measurement_data, "\n")
    f_min_measurement = 1e5
    f_max_measurement = 1.5e6
    T_min_measurement = 25
    T_max_measurement = 130
    permittivity.measurement_data = permittivity.filter_fT(permittivity.measurement_data,
                                                           f_min_measurement,
                                                           f_max_measurement,
                                                           T_min_measurement,
                                                           T_max_measurement)
    permittivity.fit_sigma()
    f = permittivity.measurement_data["f"].to_numpy()
    T = permittivity.measurement_data["T"].to_numpy()
    eps_r_meas = (permittivity.measurement_data["eps_real"] - 1j * permittivity.measurement_data[
        "eps_imag"]).to_numpy()
    delta_meas = np.degrees(np.arctan2(eps_r_meas.imag, eps_r_meas.real))
    eps_real_fit, eps_imag_fit = permittivity.fit_real_and_imaginary_part_at_f_and_T(f, T)
    eps_r_fit = eps_real_fit - 1j * eps_imag_fit
    delta_fit = np.degrees(np.arctan2(eps_r_fit.imag, eps_r_fit.real))

    if is_plot:
        plot_parity_amplitude(eps_r_meas, eps_r_fit)  # type: ignore
        plot_parity_angle(delta_meas, delta_fit)
        plot_over_frequency(T, f, eps_r_meas, eps_r_fit, delta_meas, delta_fit)  # type: ignore
        plt.show()


if __name__ == "__main__":
    fit_complex_permittivity_example(is_plot=True)
