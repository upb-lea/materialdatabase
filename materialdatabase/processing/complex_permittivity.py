"""Class to represent the data structure and load material data."""
import logging
# python libraries
from typing import List

# 3rd party libraries
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# own libraries
from materialdatabase.meta.data_enums import *
from materialdatabase.meta.config import *
from materialdatabase.processing.utils.empirical import *

logger = logging.getLogger(__name__)


class ComplexPermittivity:
    """Class to process complex permittivity data."""

    def __init__(self, df_complex_permittivity: pd.DataFrame, material: Material, measurement_setup: MeasurementSetup):
        """
        Initialize the complex permeability measurement data.

        :param df_complex_permittivity: pd.DataFrame with header ["f", "T", "eps_real", "eps_imag"]
        :param material: e.g. mdb.Material.N95
        :param measurement_setup: e.g. mdb.MeasurementSetup.TDK_MDT
        """
        self.measurement_data = df_complex_permittivity
        self.material = material
        self.measurement_setup = measurement_setup
        self.params_eps_a = None
        self.eps_a_fit_function = FitFunction.eps_abs
        self.params_eps_pv = None
        self.eps_pv_fit_function = FitFunction.eps_abs

    def fit_permittivity_magnitude(self) -> Any:
        """
        Fit the permittivity magnitude ε_abs as a function of frequency and temperature.

        This method:
          1. Computes ε_abs = sqrt(ε_real² + ε_imag²).
          2. Interpolates the magnitude to a uniform frequency grid at each temperature.
          3. Fits the interpolated data.

        :return: Fitted parameters (popt_eps_a) of the ε_abs model.
        :rtype: np.ndarray
        """
        # Step 1: Compute magnitude
        eps_a = np.sqrt(self.measurement_data["eps_real"] ** 2 + self.measurement_data["eps_imag"] ** 2)

        df = self.measurement_data.copy()
        df["eps_a"] = eps_a

        # Step 2: Interpolate to uniform frequency grid for each T
        interpolated_f: List[float] = []
        interpolated_T: List[float] = []
        interpolated_eps_a: List[float] = []

        unique_Ts = np.unique(df["T"])
        for T in unique_Ts:
            df_T = df[df["T"] == T].sort_values("f")
            f_min, f_max = df_T["f"].min(), df_T["f"].max()

            # Create evenly spaced frequency grid (e.g., same number as original points)
            f_uniform = np.linspace(f_min, f_max, len(df_T))

            # Interpolate magnitude over uniform grid
            interp_func = interp1d(df_T["f"], df_T["eps_a"], kind="linear", fill_value="extrapolate")
            eps_a_uniform = interp_func(f_uniform)

            interpolated_f.extend(f_uniform)
            interpolated_T.extend([T] * len(f_uniform))
            interpolated_eps_a.extend(eps_a_uniform)

        # Step 3: Fit to interpolated dataset
        params_eps_a, pcov_eps_a = curve_fit(
            self.eps_a_fit_function.get_function(),
            (np.array(interpolated_f), np.array(interpolated_T)),
            np.array(interpolated_eps_a),
            maxfev=int(1e6)
        )

        self.params_eps_a = params_eps_a
        return params_eps_a

    def fit_loss_angle(self) -> Any:
        """
        Fit the dielectric losses as a function of frequency and temperature.

        This method:
          1. Computes loss density p_el.
          2. Interpolates p_el to a uniform frequency grid at each temperature.
          3. Fits the interpolated.

        :return: Fitted parameters (popt_eps_pv) of the ε_abs model.
        :rtype: np.ndarray
        """
        # Step 1: Compute magnitude
        eps_angle = np.arctan(self.measurement_data["eps_imag"] / self.measurement_data["eps_real"])

        df = self.measurement_data.copy()
        df["eps_angle"] = eps_angle

        # Step 2: Interpolate to uniform frequency grid for each T
        interpolated_f: List[float] = []
        interpolated_T: List[float] = []
        interpolated_eps_angle: List[float] = []

        unique_Ts = np.unique(df["T"])
        for T in unique_Ts:
            df_T = df[df["T"] == T].sort_values("f")
            f_min, f_max = df_T["f"].min(), df_T["f"].max()

            # Create evenly spaced frequency grid (e.g., same number as original points)
            f_uniform = np.linspace(f_min, f_max, len(df_T))

            # Interpolate magnitude over uniform grid
            interp_func = interp1d(df_T["f"], df_T["eps_angle"], kind="linear", fill_value="extrapolate")
            eps_angle_uniform = interp_func(f_uniform)

            interpolated_f.extend(f_uniform)
            interpolated_T.extend([T] * len(f_uniform))
            interpolated_eps_angle.extend(eps_angle_uniform)

        # Step 3: Fit to interpolated dataset
        params_eps_pv, pcov_eps_pv = curve_fit(
            self.eps_pv_fit_function.get_function(),
            (np.array(interpolated_f), np.array(interpolated_T)),
            np.array(interpolated_eps_angle),
            maxfev=int(1e6)
        )

        self.params_eps_pv = params_eps_pv
        return params_eps_pv

    def fit_real_and_imaginary_part_at_f_and_T(self, f: float, T: float) -> tuple[float, float]:
        """
        Compute fitted real and imaginary parts of permittivity at given frequency and temperature.

        The fitted params_eps_a (magnitude) and params_eps_pv (loss angle) are used for the interpolation.
        If parameters are missing, they will be generated by running the corresponding fit methods.
        :param f: Frequency in Hz
        :param T: Temperature in °C
        :return: (eps_real, eps_imag)
        """
        # Auto-generate magnitude fit if missing
        if self.params_eps_a is None:
            logger.info("params_eps_a missing, running fit_permittivity_magnitude()...")
            self.fit_permittivity_magnitude()

        # Auto-generate loss angle fit if missing
        if self.params_eps_pv is None:
            logger.info("params_eps_pv missing, running fit_loss_angle()...")
            self.fit_loss_angle()

        # Predict magnitude ε_abs from the fitted model
        eps_a_model = self.eps_a_fit_function.get_function()
        eps_a = eps_a_model((np.array([f]), np.array([T])), *self.params_eps_a)[0]

        # Predict loss angle φ from the fitted model
        eps_pv_model = self.eps_pv_fit_function.get_function()
        phi = eps_pv_model((np.array([f]), np.array([T])), *self.params_eps_pv)[0]

        # Convert magnitude + angle → real & imaginary
        eps_real = eps_a * np.cos(phi)
        eps_imag = eps_a * np.sin(phi)

        return eps_real, eps_imag
