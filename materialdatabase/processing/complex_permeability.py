"""Class to represent the data structure and load material data."""
# python libraries
import os
from typing import Tuple, Optional

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
                 data_source: DataSource,
                 pv_fit_function: FitFunction):
        """
        Initialize the complex permeability measurement data.

        :param df_complex_permeability: pd.DataFrame with header ["f", "T", "b", "mu_real", "mu_imag"]
        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        """
        self.measurement_data = df_complex_permeability
        self.material = material
        self.data_source = data_source
        self._fitted_data: pd.DataFrame | None = None
        self.params_mu_a = None
        self.mu_a_fit_function = get_fit_function_from_setup(data_source)  # permeability fit is coupled to measurement setup
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

    @staticmethod
    def filter_fTb(df: pd.DataFrame,
                   f_min: Optional[float] = None, f_max: Optional[float] = None,
                   T_min: Optional[float] = None, T_max: Optional[float] = None,
                   b_min: Optional[float] = None, b_max: Optional[float] = None) -> pd.DataFrame:
        """
        Filter a material dataframe df for min/max frequency, temperature, and flux density.

        :param df: original material dataframe
        :param f_min: minimum frequency (default: None)
        :param f_max: maximum frequency (default: None)
        :param T_min: minimum temperature (default: None)
        :param T_max: maximum temperature (default: None)
        :param b_min: minimum flux density (default: None)
        :param b_max: maximum flux density (default: None)
        :return: filtered material dataframe
        """
        if f_min is not None:
            df = df[df['f'] >= f_min]
        if f_max is not None:
            df = df[df['f'] <= f_max]
        if T_min is not None:
            df = df[df['T'] >= T_min]
        if T_max is not None:
            df = df[df['T'] <= T_max]
        if b_min is not None:
            df = df[df['b'] >= b_min]
        if b_max is not None:
            df = df[df['b'] <= b_max]
        return df

    def fit_permeability_magnitude(self,
                                   f_min: Optional[float] = None, f_max: Optional[float] = None,
                                   T_min: Optional[float] = None, T_max: Optional[float] = None,
                                   b_min: Optional[float] = None, b_max: Optional[float] = None) -> Any:
        """
        Fit the permeability magnitude μ_abs as a function of frequency, temperature, and magnetic flux density.

        This method computes the permeability magnitude from the real and imaginary components using:
            μ_abs = sqrt(μ_real² + μ_imag²)

        It then fits this data using a predefined model function `fit_mu_abs_...(f, T, b, ...)`.

        :param f_min: measurements for lower frequencies will be excluded from fitting
        :param f_max: measurements for higher frequencies will be excluded from fitting
        :param T_min: measurements for lower temperatures will be excluded from fitting
        :param T_max: measurements for higher temperatures will be excluded from fitting
        :param b_min: measurements for lower flux densities will be excluded from fitting
        :param b_max: measurements for higher flux densities will be excluded from fitting
        :return: Fitted parameters (popt_mu_abs) of the μ_abs model.
        :rtype: np.ndarray
        """
        fit_data = self.filter_fTb(self.measurement_data,
                                   f_min=f_min, f_max=f_max,
                                   T_min=T_min, T_max=T_max,
                                   b_min=b_min, b_max=b_max)

        fit_mu_a = self.mu_a_fit_function.get_function()

        logger.info(f"\n"
                    f"Fitting of the permeability amplitude with the fit function'{self.mu_a_fit_function.value}'.\n"
                    f" Following limits are applied to the measurement data:\n"
                    f"  {f_min = }\n"
                    f"  {f_max = }\n"
                    f"  {T_min = }\n"
                    f"  {T_max = }\n"
                    f"  {b_min = }\n"
                    f"  {b_max = }\n"
                    f" Following data is used for the loss fitting:\n "
                    f"  {fit_data}")

        mu_a = np.sqrt(fit_data["mu_real"] ** 2 + fit_data["mu_imag"] ** 2)
        popt_mu_a, pcov_mu_a = curve_fit(fit_mu_a,
                                         (fit_data["f"],
                                          fit_data["T"],
                                          fit_data["b"]),
                                         mu_a, maxfev=int(1e6))
        self.params_mu_a = popt_mu_a
        return popt_mu_a

    def fit_losses(self,
                   f_min: Optional[float] = None, f_max: Optional[float] = None,
                   T_min: Optional[float] = None, T_max: Optional[float] = None,
                   b_min: Optional[float] = None, b_max: Optional[float] = None) -> Any:
        """
        Fit the magnetic power loss density p_v as a function of frequency, temperature, and magnetic flux density.

        The losses are calculated from the imaginary part of the permeability using the helper function `pv_mag()`,
        and then fitted using e.g. the Steinmetz-based `..._steinmetz_...(f, T, b, ...)`.

        :param f_min: measurements for lower frequencies will be excluded from fitting
        :param f_max: measurements for higher frequencies will be excluded from fitting
        :param T_min: measurements for lower temperatures will be excluded from fitting
        :param T_max: measurements for higher temperatures will be excluded from fitting
        :param b_min: measurements for lower flux densities will be excluded from fitting
        :param b_max: measurements for higher flux densities will be excluded from fitting
        :return: Fitted parameters (popt_pv) of the Steinmetz-based power loss model.
        :rtype: np.ndarray
        """
        log_pv_fit_function = self.pv_fit_function.get_log_function()

        fit_data = self.filter_fTb(self.measurement_data,
                                   f_min=f_min, f_max=f_max,
                                   T_min=T_min, T_max=T_max,
                                   b_min=b_min, b_max=b_max)

        logger.info(f"\n"
                    f"Fitting of the magnetic losses with the fit function'{self.pv_fit_function.value}'.\n"
                    f" Following limits are applied to the measurement data:\n"
                    f"  {f_min = }\n"
                    f"  {f_max = }\n"
                    f"  {T_min = }\n"
                    f"  {T_max = }\n"
                    f"  {b_min = }\n"
                    f"  {b_max = }\n"
                    f" Following data is used for the loss fitting:\n "
                    f"  {fit_data}")

        mu_abs = np.sqrt(fit_data["mu_real"] ** 2 + fit_data["mu_imag"] ** 2)
        pv = pv_mag(fit_data["f"].to_numpy(),
                    (-fit_data["mu_imag"] * mu_0).to_numpy(),
                    (fit_data["b"] / mu_abs / mu_0).to_numpy())
        popt_pv, pcov_pv = curve_fit(log_pv_fit_function,
                                     (fit_data["f"],
                                      fit_data["T"],
                                      fit_data["b"]),
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
    def grid2txt(df: pd.DataFrame, path: os.PathLike | str) -> None:
        """
        Export 3D permeability data to grid text file format.

        :param df: Expected DataFrame columns: [frequency, temperature, flux density, real part of permeability, imaginary part of permeability]
        :param path: path where the txt file should be stored
        """
        with open(path, "w+") as f:
            f.write("%Grid\n")

            # magnetic flux density
            b_grid = sorted(set(df[2].values.tolist()))
            f.write(str(b_grid)[1:-1] + "\n")

            # frequency
            f_grid = sorted(set(df[0].values.tolist()))
            f.write(str(f_grid)[1:-1] + "\n")

            # temperature
            T_grid = sorted(set(df[1].values.tolist()))
            f.write(str(T_grid)[1:-1] + "\n")

            f.write("%Data\n")
            mu_real = df[3].values.tolist()
            f.write(str(mu_real)[1:-1] + "\n")

            f.write("%Data\n")
            mu_imag = df[4].values.tolist()
            f.write(str(mu_imag)[1:-1])

    def to_grid(self,
                grid_frequency: npt.NDArray[Any],
                grid_temperature: npt.NDArray[Any],
                grid_flux_density: npt.NDArray[Any],
                f_min_measurement: Optional[float] = None, f_max_measurement: Optional[float] = None,
                T_min_measurement: Optional[float] = None, T_max_measurement: Optional[float] = None,
                b_min_measurement: Optional[float] = None, b_max_measurement: Optional[float] = None
                ) -> pd.DataFrame:
        """
        Export fitted permeability data (real & imaginary parts) to a txt grid file.

        :param grid_frequency: frequencies for the interpolation grid
        :param grid_temperature: temperatures for the interpolation grid
        :param grid_flux_density: magnetic flux density values for the interpolation grid
        :param f_min_measurement: measurements for lower frequencies will be excluded from fitting
        :param f_max_measurement: measurements for higher frequencies will be excluded from fitting
        :param T_min_measurement: measurements for lower temperatures will be excluded from fitting
        :param T_max_measurement: measurements for higher temperatures will be excluded from fitting
        :param b_min_measurement: measurements for lower flux densities will be excluded from fitting
        :param b_max_measurement: measurements for higher flux densities will be excluded from fitting
        """
        if self.params_pv is None:
            self.fit_losses(f_min_measurement, f_max_measurement,
                            T_min_measurement, T_max_measurement,
                            b_min_measurement, b_max_measurement)
        if self.params_mu_a is None:
            self.fit_permeability_magnitude(f_min_measurement, f_max_measurement,
                                            T_min_measurement, T_max_measurement,
                                            b_min_measurement, b_max_measurement)

        records: list[list[float]] = []
        for T in grid_temperature:
            for f in grid_frequency:
                mu_real, mu_imag = self.fit_real_and_imaginary_part_at_f_and_T(f, T, grid_flux_density)
                for i, b in enumerate(grid_flux_density):
                    records.append([f, T, b, mu_real[i], mu_imag[i]])

        df_grid: pd.DataFrame = pd.DataFrame(records, columns=[0, 1, 2, 3, 4])
        return df_grid

