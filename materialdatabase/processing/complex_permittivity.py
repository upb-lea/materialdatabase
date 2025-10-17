"""Class to represent the data structure and load material data."""
# python libraries
import os

# 3rd party libraries
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

# own libraries
from materialdatabase.meta.data_enums import *
from materialdatabase.meta.config import *
from materialdatabase.processing.utils.empirical import *

logger = logging.getLogger(__name__)


class ComplexPermittivity:
    """Class to process complex permittivity data."""

    def __init__(self, df_complex_permittivity: pd.DataFrame, material: Material, data_source: DataSource):
        """
        Initialize the complex permeability measurement data.

        :param df_complex_permittivity: pd.DataFrame with header ["f", "T", "eps_real", "eps_imag"]
        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        """
        self.measurement_data = df_complex_permittivity
        self.material = material
        self.data_source = data_source
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
        interpolated_f: list[float] = []
        interpolated_T: list[float] = []
        interpolated_eps_a: list[float] = []

        unique_Ts = np.unique(df["T"])
        for T in unique_Ts:
            df_T = df[df["T"] == T].sort_values("f")
            f_min, f_max = df_T["f"].min(), df_T["f"].max()

            # Create evenly spaced frequency grid (e.g., same number as original points or a fixed number like 20)
            # f_uniform = np.linspace(f_min, f_max, len(df_T))
            f_uniform = np.linspace(f_min, f_max, 20)

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
        interpolated_f: list[float] = []
        interpolated_T: list[float] = []
        interpolated_eps_angle: list[float] = []

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

    @staticmethod
    def grid2txt(df: pd.DataFrame, path: os.PathLike | str) -> None:
        """
        Export 2D permittivity data to grid text file format.

        :param df: Expected DataFrame columns: [frequency, temperature, real part of permittivity, imaginary part of permittivity]
        :param path: path where the txt file should be stored
        """
        with open(path, "w+") as f:
            f.write("%Grid\n")

            # frequency
            f_grid = sorted(set(df["f"].values.tolist()))
            f.write(str(f_grid)[1:-1] + "\n")

            # temperature
            T_grid = sorted(set(df["T"].values.tolist()))
            f.write(str(T_grid)[1:-1] + "\n")

            f.write("%Data\n")
            eps_real = df["eps_real"].values.tolist()
            f.write(str(eps_real)[1:-1] + "\n")

            f.write("%Data\n")
            eps_imag = df["eps_imag"].values.tolist()
            f.write(str(eps_imag)[1:-1])

    def to_grid(self,
                grid_frequency: npt.NDArray[Any],
                grid_temperature: npt.NDArray[Any]) -> pd.DataFrame:
        """
        Export fitted permittivity data (real & imaginary parts) to a txt grid file.

        :param grid_frequency: frequencies for the interpolation grid
        :param grid_temperature: temperatures for the interpolation grid
        """
        if self.params_eps_a is None:
            self.fit_permittivity_magnitude()
        if self.params_eps_pv is None:
            self.fit_loss_angle()

        records: list[list[float]] = []
        for T in grid_temperature:
            for f in grid_frequency:
                eps_real, eps_imag = self.fit_real_and_imaginary_part_at_f_and_T(f, T)
                records.append([f, T, eps_real, eps_imag])

        df_grid: pd.DataFrame = pd.DataFrame(records, columns=["f", "T", "eps_real", "eps_imag"])
        return df_grid

    @staticmethod
    def plot_grid(df: pd.DataFrame,
                  save_path: str | Path,
                  no_levels: int = 100,
                  f_min: float | None = None, f_max: float | None = None,
                  T_min: float | None = None, T_max: float | None = None) -> None:
        """
        Plot |ε| and phase(ε) as contour maps vs. f (kHz) and T (°C) with shared color scales for magnitude and phase.

        :param df: complex permittivity data as pandas DataFrame
        :param save_path: path where the plot should be saved
        :param no_levels: number of levels to show
        :param f_min: minimum frequency
        :param f_max: maximum frequency
        :param T_min: minimum temperature
        :param T_max: maximum temperature
        """
        # -------------------------
        # Matplotlib settings
        # -------------------------
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10.0,
            "text.latex.preamble": r"""
                \usepackage{amsmath}
                \usepackage{upgreek}
                \usepackage{bm}
            """
        })

        # --- Sanity checks ---
        required_columns = ['f', 'T', 'eps_real', 'eps_imag']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # --- Filter by optional limits ---
        df_plot = df.copy()
        if f_min is not None:
            df_plot = df_plot[df_plot['f'] >= f_min]
        if f_max is not None:
            df_plot = df_plot[df_plot['f'] <= f_max]
        if T_min is not None:
            df_plot = df_plot[df_plot['T'] >= T_min]
        if T_max is not None:
            df_plot = df_plot[df_plot['T'] <= T_max]
        if df_plot.empty:
            raise ValueError("No data remaining after applying filter limits.")

        # --- Compute magnitude and phase ---
        eps_abs = np.sqrt(df_plot['eps_real'] ** 2 + df_plot['eps_imag'] ** 2)
        eps_phi = np.rad2deg(np.arctan2(df_plot['eps_imag'], df_plot['eps_real']))

        # --- Common grid for interpolation ---
        f_vals = np.linspace(df_plot['f'].min(), df_plot['f'].max(), 200)
        T_vals = np.linspace(df_plot['T'].min(), df_plot['T'].max(), 200)
        grid_f, grid_T = np.meshgrid(f_vals, T_vals)
        grid_f_kHz = grid_f * 1e-3  # Hz → kHz

        # --- Interpolation ---
        grid_eps_abs = griddata((df_plot['f'], df_plot['T']), eps_abs, (grid_f, grid_T), method='cubic')
        grid_eps_phi = griddata((df_plot['f'], df_plot['T']), eps_phi, (grid_f, grid_T), method='cubic')

        # --- Shared color scales ---
        abs_min, abs_max = np.nanmin(grid_eps_abs), np.nanmax(grid_eps_abs)
        phi_min, phi_max = np.nanmin(grid_eps_phi), np.nanmax(grid_eps_phi)
        levels_abs = np.linspace(abs_min, abs_max, no_levels)
        levels_phi = np.linspace(phi_min, phi_max, no_levels)

        # --- Figure setup with GridSpec ---
        cm = 1 / 2.54
        fig = plt.figure(figsize=(5 * cm, 9 * cm))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], wspace=0.15, hspace=0.15)
        ax_abs = fig.add_subplot(gs[0, 0])
        ax_phi = fig.add_subplot(gs[1, 0])
        cax_abs = fig.add_subplot(gs[0, 1])
        cax_phi = fig.add_subplot(gs[1, 1])

        # --- Contour plots ---
        cont_abs = ax_abs.contourf(grid_f_kHz, grid_T, grid_eps_abs, levels=levels_abs, cmap='plasma',
                                   vmin=abs_min, vmax=abs_max)
        cont_phi = ax_phi.contourf(grid_f_kHz, grid_T, grid_eps_phi, levels=levels_phi, cmap='plasma',
                                   vmin=phi_min, vmax=phi_max)

        # --- Labels ---
        ax_abs.set_ylabel(r"$T$ / °C")
        ax_phi.set_ylabel(r"$T$ / °C")
        ax_phi.set_xlabel(r"$f$ / kHz")

        # --- Colorbars ---
        for cont, cax, label in zip([cont_abs, cont_phi],
                                    [cax_abs, cax_phi],
                                    [r"$|\tilde{\varepsilon}_r|$", r"$\zeta_{\tilde{\varepsilon}}$ / °"], strict=False):
            cbar = fig.colorbar(cont, cax=cax)
            cbar.set_label(label)
            cbar.locator = MaxNLocator(nbins=5)
            cbar.update_ticks()

        # --- Simplify ticks ---
        for a in [ax_abs, ax_phi]:
            a.xaxis.set_major_locator(MaxNLocator(nbins=4))
            a.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # --- Remove upper plot x-axis tick labels ---
        ax_abs.set_xticklabels([])  # <-- hide x-axis tick labels on top plot

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
