"""Example script to demonstrate processing and fitting complex permittivity data using the ComplexPermittivity class from the materialdatabase package."""

import logging
import numpy as np
from matplotlib import pyplot as plt
import materialdatabase as mdb

# Configure logging to show info level messages
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def plot_permittivity_magnitude_fit(complex_perm):
    """
    Plot measured vs fitted permittivity magnitude (ε_abs) for all (f, T) combinations.

    Parameters
    ----------
    complex_perm : ComplexPermittivity
        Instance containing measurement data and fitted parameters.
    """
    eps_a_measured = np.sqrt(complex_perm.measurement_data["eps_real"] ** 2 + complex_perm.measurement_data["eps_imag"] ** 2)

    fit_func = complex_perm.eps_a_fit_function.get_function()
    eps_a_fitted = fit_func(
        (complex_perm.measurement_data["f"], complex_perm.measurement_data["T"]),
        *complex_perm.params_eps_a
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(eps_a_measured, eps_a_fitted, alpha=0.7, label="Fitted vs Measured")
    plt.plot([eps_a_measured.min(), eps_a_measured.max()],
             [eps_a_measured.min(), eps_a_measured.max()],
             'r--', label="Ideal Fit (y = x)")
    plt.xlabel("Measured ε_abs")
    plt.ylabel("Fitted ε_abs")
    plt.title("Permittivity Magnitude: Measured vs. Fitted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_permittivity_vs_frequency(complex_perm):
    """
    Plot measured and fitted permittivity magnitude (ε_abs) vs frequency for each temperature.

    Each temperature is displayed in its own subplot.
    Parameters
    ----------
    complex_perm : ComplexPermittivity
        Instance containing measurement data and fitted parameters.
    """
    unique_temps = np.unique(complex_perm.measurement_data["T"])

    eps_a_measured = np.sqrt(complex_perm.measurement_data["eps_real"] ** 2 + complex_perm.measurement_data["eps_imag"] ** 2)

    fit_func = complex_perm.eps_a_fit_function.get_function()
    eps_a_fitted = fit_func(
        (complex_perm.measurement_data["f"], complex_perm.measurement_data["T"]),
        *complex_perm.params_eps_a
    )

    n_temps = len(unique_temps)
    fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)

    if n_temps == 1:
        axes = [axes]  # ensure iterable

    for ax, temp in zip(axes, unique_temps, strict=False):
        mask = complex_perm.measurement_data["T"] == temp
        f_vals = complex_perm.measurement_data["f"][mask]
        measured_vals = eps_a_measured[mask]
        fitted_vals = eps_a_fitted[mask]

        ax.plot(f_vals, measured_vals, 'o', label="Measured")
        ax.plot(f_vals, fitted_vals, '-', label="Fitted")
        ax.set_title(f"Temperature = {temp} °C")
        ax.set_ylabel("ε_abs")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


def plot_permittivity_loss_angle_fit(complex_perm):
    """
    Plot measured vs fitted loss angle (δ) for all (f, T) combinations.

    Parameters
    ----------
    complex_perm : ComplexPermittivity
        Instance containing measurement data and fitted parameters.
    """
    # Compute measured loss angle δ = atan(ε_imag / ε_real), convert to degrees
    delta_measured = np.degrees(np.arctan2(complex_perm.measurement_data["eps_imag"], complex_perm.measurement_data["eps_real"]))

    fit_func = complex_perm.eps_pv_fit_function.get_function()
    delta_fitted = np.degrees(fit_func(
        (complex_perm.measurement_data["f"], complex_perm.measurement_data["T"]),
        *complex_perm.params_eps_pv
    ))

    plt.figure(figsize=(8, 6))
    plt.scatter(delta_measured, delta_fitted, alpha=0.7, label="Fitted vs Measured")
    plt.plot([delta_measured.min(), delta_measured.max()],
             [delta_measured.min(), delta_measured.max()],
             'r--', label="Ideal Fit (y = x)")
    plt.xlabel("Measured Loss Angle δ (°)")
    plt.ylabel("Fitted Loss Angle δ (°)")
    plt.title("Loss Angle: Measured vs. Fitted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_permittivity_loss_angle_vs_frequency(complex_perm):
    """
    Plot measured and fitted loss angle (δ) vs frequency for each temperature.

    Each temperature is displayed in its own subplot.

    Parameters
    ----------
    complex_perm : ComplexPermittivity
        Instance containing measurement data and fitted parameters.
    """
    unique_temps = np.unique(complex_perm.measurement_data["T"])

    delta_measured = np.degrees(np.arctan2(complex_perm.measurement_data["eps_imag"], complex_perm.measurement_data["eps_real"]))

    fit_func = complex_perm.eps_pv_fit_function.get_function()
    delta_fitted = np.degrees(fit_func(
        (complex_perm.measurement_data["f"], complex_perm.measurement_data["T"]),
        *complex_perm.params_eps_pv
    ))

    n_temps = len(unique_temps)
    fig, axes = plt.subplots(n_temps, 1, figsize=(8, 3 * n_temps), sharex=True)

    if n_temps == 1:
        axes = [axes]  # ensure iterable

    for ax, temp in zip(axes, unique_temps, strict=False):
        mask = complex_perm.measurement_data["T"] == temp
        f_vals = complex_perm.measurement_data["f"][mask]
        measured_vals = delta_measured[mask]
        fitted_vals = delta_fitted[mask]

        ax.plot(f_vals, measured_vals, 'o', label="Measured")
        ax.plot(f_vals, fitted_vals, '-', label="Fitted")
        ax.set_title(f"Temperature = {temp} °C")
        ax.set_ylabel("Loss Angle δ (°)")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


def main():
    """Fit the permittivity."""
    # Initialize the material database
    mdb_data = mdb.Data()

    # Load permittivity data for a specific material and setup
    eps_N49 = mdb_data.get_complex_permittivity(
        material=mdb.Material._3F46,
        measurement_setup=mdb.MeasurementSetup.LEA_MTB
    )

    print("Exemplary complex permittivity data:")
    print(eps_N49.measurement_data, "\n")

    # Fit permittivity magnitude ε_abs
    eps_N49.fit_permittivity_magnitude()
    eps_N49.fit_loss_angle()

    # Plot measured vs fitted magnitude
    plot_permittivity_magnitude_fit(eps_N49)
    plot_permittivity_loss_angle_fit(eps_N49)

    # Plot measured and fitted ε_abs vs frequency for each temperature
    plot_permittivity_vs_frequency(eps_N49)
    plot_permittivity_loss_angle_vs_frequency(eps_N49)


if __name__ == "__main__":
    main()
