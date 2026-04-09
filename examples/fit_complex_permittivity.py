"""Example script to demonstrate processing and fitting complex permittivity data using the ComplexPermittivity class from the materialdatabase package."""

import logging
import numpy as np
from matplotlib import pyplot as plt
import materialdatabase as mdb

# Configure logging to show info level messages
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


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

    # Fit permittivity magnitude ε_abs
    f_min_measurement = 1e5
    f_max_measurement = 1.5e6
    T_min_measurement = 25
    T_max_measurement = 130

    permittivity.measurement_data = permittivity.filter_fT(permittivity.measurement_data,
                                                           f_min_measurement,
                                                           f_max_measurement,
                                                           T_min_measurement,
                                                           T_max_measurement)

    # Fit permittivity magnitude
    permittivity.fit_sigma()

    # Frequency and temperatures at which measurement data is available
    f = permittivity.measurement_data["f"].to_numpy()
    T = permittivity.measurement_data["T"].to_numpy()

    # Compute measured amplitudes and loss angles
    eps_r_meas = (permittivity.measurement_data["eps_real"] - 1j * permittivity.measurement_data["eps_imag"]).to_numpy()
    delta_meas = np.degrees(np.arctan2(eps_r_meas.imag, eps_r_meas.real))

    # Fit data
    eps_real_fit, eps_imag_fit = permittivity.fit_real_and_imaginary_part_at_f_and_T(f, T)

    # Compute fitted amplitudes and loss angles
    eps_r_fit = eps_real_fit - 1j * eps_imag_fit
    delta_fit = np.degrees(np.arctan2(eps_r_fit.imag, eps_r_fit.real))

    def mre(x, x_est):
        return np.mean(abs((x - x_est) / x))

    if is_plot:
        # ---------------
        # Parity Plots
        # ---------------
        # Amplitude
        print(f"MRE (amplitude): {np.round(100 * mre(abs(eps_r_meas), abs(eps_r_fit)), decimals=2)} %")
        plt.figure(figsize=(8, 6))
        plt.scatter(abs(eps_r_meas), np.abs(eps_r_fit), alpha=0.7, label="Fitted vs Measured")
        plt.plot([abs(eps_r_meas).min(), abs(eps_r_meas).max()],
                 [abs(eps_r_meas).min(), abs(eps_r_meas).max()],
                 'r--', label="Ideal Fit (y = x)")
        plt.xlabel("Measured ε_abs")
        plt.ylabel("Fitted ε_abs")
        plt.title("Permittivity Magnitude: Measured vs. Fitted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Loss angle
        print(f"MRE (angle): {np.round(100 * mre(delta_meas, delta_fit), decimals=2)} %")
        plt.figure(figsize=(8, 6))
        plt.scatter(delta_meas, delta_fit, alpha=0.7, label="Fitted vs Measured")
        plt.plot([delta_meas.min(), delta_meas.max()],
                 [delta_meas.min(), delta_meas.max()],
                 'r--', label="Ideal Fit (y = x)")
        plt.xlabel("Measured Loss Angle δ (°)")
        plt.ylabel("Fitted Loss Angle δ (°)")
        plt.title("Loss Angle: Measured vs. Fitted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # ---------------
        # over frequency (at temperature levels)
        # ---------------
        unique_temps = np.unique(T)
        n_temps = len(unique_temps)

        # Amplitude
        fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)

        if n_temps == 1:
            axes = [axes]  # ensure iterable

        for ax, temp in zip(axes, unique_temps, strict=False):
            mask = T == temp
            f_vals = f[mask]
            measured_vals = abs(eps_r_meas)[mask]
            fitted_vals = np.abs(eps_r_fit)[mask]

            ax.plot(f_vals, measured_vals, 'o', label="Measured")
            ax.plot(f_vals, fitted_vals, 'x', label="Fitted")
            ax.set_title(f"Temperature = {temp} °C")
            ax.set_ylabel("Fitted ε_abs")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()

        # Loss angle
        fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)

        if n_temps == 1:
            axes = [axes]  # ensure iterable

        for ax, temp in zip(axes, unique_temps, strict=False):
            mask = T == temp
            f_vals = f[mask]
            measured_vals = delta_meas[mask]
            fitted_vals = delta_fit[mask]

            ax.plot(f_vals, measured_vals, 'o', label="Measured")
            ax.plot(f_vals, fitted_vals, 'x', label="Fitted")
            ax.set_title(f"Temperature = {temp} °C")
            ax.set_ylabel("Loss Angle δ (°)")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()

        # show all plots
        plt.show()


if __name__ == "__main__":
    fit_complex_permittivity_example(is_plot=True)
