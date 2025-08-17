"""Class to represent the data structure and load material data."""
# python libraries
import os
from typing import Any, Tuple

# 3rd party libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# own libraries
from materialdatabase.meta.data_enums import *
from materialdatabase.meta.config import *
from materialdatabase.meta.mapping import *
from materialdatabase.processing.utils.empirical import *
from materialdatabase.processing.utils.physic import pv_mag, mu_imag_from_pv
from materialdatabase.processing.utils.constants import mu_0

logger = logging.getLogger(__name__)


class ComplexPermeability:
    """Class to process complex permeability data."""

    def __init__(self,
                 df_complex_permeability: pd.DataFrame,
                 material: Material,
                 measurement_setup: MeasurementSetup,
                 pv_fit_function: FitFunction):
        """
        Initialize the complex permeability measurement data.

        :param df_complex_permeability: pd.DataFrame with header ["f", "T", "b", "mu_real", "mu_imag"]
        :param material: e.g. mdb.Material.N95
        :param measurement_setup: e.g. mdb.MeasurementSetup.TDK_MDT
        """
        self.measurement_data = df_complex_permeability
        self.material = material
        self.measurement_setup = measurement_setup
        self._fitted_data: pd.DataFrame | None = None
        self.params_mu_a = None
        self.mu_a_fit_function = get_fit_function_from_setup(measurement_setup)  # permeability fit is coupled to measurement setup
        self.params_pv = None
        self.pv_fit_function = pv_fit_function

    def get_measurement_bounds(self) -> dict:
        """
        Return the min and max values of frequency (f), temperature (T), and flux density (b) from the measurement data.

        :return: Dictionary with structure:
                 {
                    "f": (f_min, f_max),
                    "T": (T_min, T_max),
                    "b": (b_min, b_max)
                 }
        """
        df = self.measurement_data

        bounds = {
            "f": (df["f"].min(), df["f"].max()),
            "T": (df["T"].min(), df["T"].max()),
            "b": (df["b"].min(), df["b"].max())
        }

        return bounds

    def generate_1d_interpolation_arrays(self, f_points: int = 20, T_points: int = 20, b_points: int = 20) -> dict:
        """
        Generate 1D np.linspace arrays for each of the measurement parameters.

        frequency (f), temperature (T), and flux density (b) are based on their measured min and max values.


        :param f_points: Number of points in the frequency array (default: 20)
        :param T_points: Number of points in the temperature array (default: 20)
        :param b_points: Number of points in the flux density array (default: 20)
        :return: Dictionary with keys "f", "T", and "b", each mapped to a 1D np.ndarray
                 e.g. {
                     "f": np.ndarray of shape (f_points,),
                     "T": np.ndarray of shape (T_points,),
                     "b": np.ndarray of shape (b_points,)
                 }
        """
        bounds = self.get_measurement_bounds()

        grid = {
            "f": np.linspace(bounds["f"][0], bounds["f"][1], f_points),
            "T": np.linspace(bounds["T"][0], bounds["T"][1], T_points),
            "b": np.linspace(bounds["b"][0], bounds["b"][1], b_points),
        }

        return grid

    def fit_permeability_magnitude(self) -> Any:
        """
        Fit the permeability magnitude μ_abs as a function of frequency, temperature, and magnetic flux density.

        This method computes the permeability magnitude from the real and imaginary components using:
            μ_abs = sqrt(μ_real² + μ_imag²)

        It then fits this data using a predefined model function `fit_mu_abs_...(f, T, b, ...)`.

        :return: Fitted parameters (popt_mu_abs) of the μ_abs model.
        :rtype: np.ndarray
        """
        fit_mu_a = self.mu_a_fit_function.get_function()

        mu_a = np.sqrt(self.measurement_data["mu_real"] ** 2 + self.measurement_data["mu_imag"] ** 2)
        popt_mu_a, pcov_mu_a = curve_fit(fit_mu_a,
                                         (self.measurement_data["f"],
                                          self.measurement_data["T"],
                                          self.measurement_data["b"]),
                                         mu_a, maxfev=int(1e6))
        self.params_mu_a = popt_mu_a
        return popt_mu_a

    def fit_losses(self) -> Any:
        """
        Fit the magnetic power loss density p_v as a function of frequency, temperature, and magnetic flux density.

        The losses are calculated from the imaginary part of the permeability using the helper function `pv_mag()`,
        and then fitted using e.g. the Steinmetz-based `..._steinmetz_...(f, T, b, ...)`.

        :return: Fitted parameters (popt_pv) of the Steinmetz-based power loss model.
        :rtype: np.ndarray
        """
        log_pv_fit_function = self.pv_fit_function.get_log_function()
        pv_fit_function = self.pv_fit_function.get_function()

        mu_abs = np.sqrt(self.measurement_data["mu_real"] ** 2 + self.measurement_data["mu_imag"] ** 2)
        pv = pv_mag(self.measurement_data["f"].to_numpy(),
                    (-self.measurement_data["mu_imag"] * mu_0).to_numpy(),
                    (self.measurement_data["b"] / mu_abs / mu_0).to_numpy())
        popt_pv, pcov_pv = curve_fit(log_pv_fit_function,
                                     (self.measurement_data["f"],
                                      self.measurement_data["T"],
                                      self.measurement_data["b"]),
                                     np.log(pv), maxfev=100000)
        self.params_pv = popt_pv
        return popt_pv

    def fit_real_and_imaginary_part_at_f_and_T(
            self,
            f_op: float,
            T_op: float,
            b_vals: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Fit permeability and losses for a given material and return real/imaginary parts.

        :param f_op: Operating frequency in Hz
        :param T_op: Operating temperature in °C
        :param b_vals: 1D np.ndarray of magnetic flux density values in T
        :return: Tuple (mu_real, mu_imag) of fitted real and imaginary parts of permeability
        """
        assert isinstance(b_vals, np.ndarray), "b_vals must be a NumPy array"

        if self.params_pv is None:
            self.fit_losses()
        if self.params_mu_a is None:
            self.fit_permeability_magnitude()

        # Avoid division by zero
        if b_vals[0] == 0:
            b_vals = b_vals.copy()
            b_vals[0] = 1e-6

        # Fit permeability magnitude μₐ(f, T, B)
        mu_a = self.mu_a_fit_function.get_function()((f_op, T_op, b_vals), *self.params_mu_a)

        # Fit specific power loss using provided model
        pv_vals = self.pv_fit_function.get_function()((f_op, T_op, b_vals), *self.params_pv)

        # Compute excitation field strength H = B / (μₐ * μ₀)
        h_vals = b_vals / mu_a / mu_0

        # Compute imaginary part of permeability from losses
        mu_imag = -mu_imag_from_pv(f_op, h_vals, pv_vals) / mu_0

        # Compute real part using magnitude and imaginary part
        mu_real = np.sqrt(np.maximum(mu_a ** 2 - mu_imag ** 2, 0))

        return np.array(mu_real), np.array(mu_imag)

    @staticmethod
    def txt2grid3d(df: pd.DataFrame, path: os.PathLike | str) -> None:
        """
        Export 3D permeability data to grid text file format.

        :param df: Expected DataFrame columns: [frequency, temperature, flux density, real part of permeability, imaginary part of permeability]
        :param path: path where the txt file should be stored
        """
        with open(path, "w+") as f:
            f.write("%Grid\n")

            # magnetic flux density
            b_grid: list[float] = sorted(set(df[2].values.tolist()))
            f.write(str(b_grid)[1:-1] + "\n")

            # frequency
            f_grid: list[float] = sorted(set(df[0].values.tolist()))
            f.write(str(f_grid)[1:-1] + "\n")

            # temperature
            T_grid: list[float] = sorted(set(df[1].values.tolist()))
            f.write(str(T_grid)[1:-1] + "\n")

            f.write("%Data\n")
            mu_real: list[float] = df[3].values.tolist()
            f.write(str(mu_real)[1:-1] + "\n")

            f.write("%Data\n")
            mu_imag: list[float] = df[4].values.tolist()
            f.write(str(mu_imag)[1:-1])

    def export_to_txt(self, path: str | os.PathLike, frequencies: npt.NDArray[Any], temperatures: npt.NDArray[Any],
                      b_vals: npt.NDArray[Any]) -> None:
        """
        Export fitted permeability data (real & imaginary parts) to a txt grid file.

        :param path: path to exported txt-file
        :param frequencies: frequencies for the interpolation grid
        :param temperatures: temperatures for the interpolation grid
        :param b_vals: magnetic flux density values for the interpolation grid
        """
        if self.params_pv is None:
            self.fit_losses()
        if self.params_mu_a is None:
            self.fit_permeability_magnitude()

        records: list[list[float]] = []
        for T in temperatures:
            for f in frequencies:
                mu_real, mu_imag = self.fit_real_and_imaginary_part_at_f_and_T(f, T, b_vals)
                for i, b in enumerate(b_vals):
                    records.append([f, T, b, mu_real[i], mu_imag[i]])

        df_export: pd.DataFrame = pd.DataFrame(records, columns=[0, 1, 2, 3, 4])
        self.txt2grid3d(df_export, path)
