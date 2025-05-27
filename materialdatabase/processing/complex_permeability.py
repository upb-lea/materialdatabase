"""Class to represent the data structure and load material data."""
import logging
# python libraries

# 3rd party libraries
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

# own libraries
from materialdatabase.meta.data_enums import *
from materialdatabase.meta.config import *
from materialdatabase.processing.utils.empirical import *
from materialdatabase.processing.utils.physic import pv_mag
from materialdatabase.processing.utils.constants import mu_0

logger = logging.getLogger(__name__)


class ComplexPermeability:
    """Class to process complex permeability data."""

    def __init__(self, df_complex_permeability: pd.DataFrame, material: Material, measurement_setup: MeasurementSetup):
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
        self._mu_real_interp: RegularGridInterpolator | None = None
        self._mu_imag_interp: RegularGridInterpolator | None = None

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

    def fit_permeability_amplitude(self):
        """
        Fit the amplitude permeability μ_abs as a function of frequency, temperature, and magnetic flux density.

        This method computes the absolute permeability from the real and imaginary components using:
            μ_abs = sqrt(μ_real² + μ_imag²)

        It then fits this data using a predefined model function `fit_mu_abs_...(f, T, b, ...)`.

        :return: Fitted parameters (popt_mu_abs) of the μ_abs model.
        :rtype: np.ndarray
        """
        mu_abs = np.sqrt(self.measurement_data["mu_real"] ** 2 + self.measurement_data["mu_imag"] ** 2)
        popt_mu_abs, pcov_mu_abs = curve_fit(fit_mu_abs_qT,
                                             (self.measurement_data["f"],
                                              self.measurement_data["T"],
                                              self.measurement_data["b"]),
                                             mu_abs, maxfev=100000)
        return popt_mu_abs

    def fit_losses(self, log_pv_fit_function):
        """
        Fit the magnetic power loss density p_v as a function of frequency, temperature, and magnetic flux density.

        The losses are calculated from the imaginary part of the permeability using the helper function `pv_mag()`,
        and then fitted using e.g. the Steinmetz-based `..._steinmetz_...(f, T, b, ...)`.

        :param log_pv_fit_function: e.g. mdb.log_steinmetz_qT, mdb.log_enhanced_steinmetz_qT
        :return: Fitted parameters (popt_pv) of the Steinmetz-based power loss model.
        :rtype: np.ndarray
        """
        mu_abs = np.sqrt(self.measurement_data["mu_real"] ** 2 + self.measurement_data["mu_imag"] ** 2)
        pv = pv_mag(self.measurement_data["f"],
                    -self.measurement_data["mu_imag"] * mu_0,
                    self.measurement_data["b"] / mu_abs / mu_0)
        popt_pv, pcov_pv = curve_fit(log_pv_fit_function,
                                     (self.measurement_data["f"],
                                      self.measurement_data["T"],
                                      self.measurement_data["b"]),
                                     np.log(pv), maxfev=100000)
        return popt_pv
