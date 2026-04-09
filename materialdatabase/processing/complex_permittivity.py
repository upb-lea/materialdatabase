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
from materialdatabase.processing.utils.physic import eps_r_from_sigma, sigma_from_eps_r
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
        self.params_sigma_real = None
        self.sigma_real_fit_function = FitFunction.sigma
        self.params_sigma_imag = None
        self.sigma_imag_fit_function = FitFunction.sigma

    @staticmethod
    def filter_fT(df: pd.DataFrame,
                  f_min: float | None = None, f_max: float | None = None,
                  T_min: float | None = None, T_max: float | None = None) -> pd.DataFrame:
        """
        Filter a material dataframe df for min/max frequency, temperature, and flux density.

        :param df: original material dataframe
        :param f_min: minimum frequency (default: None)
        :param f_max: maximum frequency (default: None)
        :param T_min: minimum temperature (default: None)
        :param T_max: maximum temperature (default: None)
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
        return df

    def fit_sigma(self,
                  f_min: float | None = None, f_max: float | None = None,
                  T_min: float | None = None, T_max: float | None = None
                  ) -> Any:
        """
        Fit the complex conductivity sigma as a function of frequency and temperature.

        This method:
          1. Computes conductivity sigma from permittivity eps.
          2. For each probe: Interpolates the real/imaginary parts to a uniform frequency grid at each temperature.
          3. Fits the assembled interpolated data for real/imaginary parts.

        :param f_min: measurements for lower frequencies will be excluded from fitting
        :param f_max: measurements for higher frequencies will be excluded from fitting
        :param T_min: measurements for lower temperatures will be excluded from fitting
        :param T_max: measurements for higher temperatures will be excluded from fitting
        :return: Fitted parameters of the conductivity real and imaginary parts.
        :rtype: np.ndarray
        """
        fit_data = self.filter_fT(self.measurement_data,
                                  f_min=f_min, f_max=f_max,
                                  T_min=T_min, T_max=T_max)

        # Step 1: Compute conductivity sigma
        f = fit_data["f"].to_numpy()

        # assemble complex permittivity from the measurement data using the negative sign convention
        eps_complex = fit_data["eps_real"].to_numpy(dtype=float) - 1j * fit_data["eps_imag"].to_numpy(dtype=float)

        # compute the complex conductivity sigma, representing the measurement data
        sigma_complex = sigma_from_eps_r(f, eps_complex)

        # split real and imaginary parts for simplified fitting
        fit_data["sigma_real"] = np.real(sigma_complex)
        fit_data["sigma_imag"] = np.imag(sigma_complex)

        # Step 2: Interpolate to uniform frequency grid for each T
        interpolated_f: list[float] = []
        interpolated_T: list[float] = []
        interpolated_sigma_real: list[float] = []
        interpolated_sigma_imag: list[float] = []

        unique_probes = np.unique(fit_data["probe"])
        for probe in unique_probes:
            df_probe = fit_data[fit_data["probe"] == probe]

            unique_Ts = np.unique(df_probe["T"])
            for T in unique_Ts:
                df_T = df_probe[df_probe["T"] == T].sort_values("f")

                # Create evenly spaced frequency grid (e.g., same number as original points or a fixed number like 20)
                f_uniform = np.linspace(df_T["f"].min(), df_T["f"].max(), 20)

                # Interpolate sigma over uniform grid
                interp_sigma_real_func = interp1d(df_T["f"], df_T["sigma_real"], kind="linear", fill_value="extrapolate")
                interp_sigma_imag_func = interp1d(df_T["f"], df_T["sigma_imag"], kind="linear", fill_value="extrapolate")
                sigma_real_uniform = interp_sigma_real_func(f_uniform)
                sigma_imag_uniform = interp_sigma_imag_func(f_uniform)

                interpolated_f.extend(f_uniform)
                interpolated_T.extend([T] * len(f_uniform))
                interpolated_sigma_real.extend(sigma_real_uniform)
                interpolated_sigma_imag.extend(sigma_imag_uniform)

        # Step 3: Fit to interpolated dataset
        params_sigma_real, pcov_sigma_real = curve_fit(
            self.sigma_real_fit_function.get_function(),
            (np.array(interpolated_f), np.array(interpolated_T)),
            np.array(interpolated_sigma_real),
            maxfev=int(1e6)
        )

        params_sigma_imag, pcov_sigma_imag = curve_fit(
            self.sigma_imag_fit_function.get_function(),
            (np.array(interpolated_f), np.array(interpolated_T)),
            np.array(interpolated_sigma_imag),
            maxfev=int(1e6)
        )

        # store fit parameters internally
        self.params_sigma_real = params_sigma_real
        self.params_sigma_imag = params_sigma_imag

        return params_sigma_real, params_sigma_imag

    def fit_real_and_imaginary_part_at_f_and_T(self, f: float | np.ndarray, T: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Compute fitted real and imaginary parts of permittivity at given frequency and temperature.

        The fitted params_eps_a (magnitude) and params_eps_pv (loss angle) are used for the interpolation.
        If parameters are missing, they will be generated by running the corresponding fit methods.
        :param f: Frequency in Hz
        :param T: Temperature in °C
        :return: (eps_real, -eps_imag) (negative sign convention of the imaginary part)
        """
        # Auto-generate magnitude fit if missing
        if self.params_sigma_real is None or self.params_sigma_imag is None:
            logger.info("params_eps_a missing, running fit_permittivity_magnitude()...")
            self.fit_sigma()

        # Predict conductivity real part the fitted model
        sigma_real_model = self.sigma_real_fit_function.get_function()
        sigma_real = sigma_real_model((np.array([f]), np.array([T])), *self.params_sigma_real)[0]

        # Predict conductivity imaginary part from the fitted model
        sigma_imag_model = self.sigma_imag_fit_function.get_function()
        sigma_imag = sigma_imag_model((np.array([f]), np.array([T])), *self.params_sigma_imag)[0]

        # assemble complex conductivity
        sigma_complex = sigma_real + 1j * sigma_imag

        # Convert complex conductivity to permittivity
        eps_complex = eps_r_from_sigma(f, sigma_complex)

        return eps_complex.real, -eps_complex.imag

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
                grid_temperature: npt.NDArray[Any],
                f_min_measurement: float | None = None, f_max_measurement: float | None = None,
                T_min_measurement: float | None = None, T_max_measurement: float | None = None
                ) -> pd.DataFrame:
        """
        Export fitted permittivity data (real & imaginary parts) to a txt grid file.

        :param f_min_measurement: measurements for lower frequencies will be excluded from fitting
        :param f_max_measurement: measurements for higher frequencies will be excluded from fitting
        :param T_min_measurement: measurements for lower temperatures will be excluded from fitting
        :param T_max_measurement: measurements for higher temperatures will be excluded from fitting
        :param grid_frequency: frequencies for the interpolation grid
        :param grid_temperature: temperatures for the interpolation grid
        """
        if self.params_sigma_real is None or self.params_sigma_imag is None:
            self.fit_sigma(f_min_measurement, f_max_measurement,
                           T_min_measurement, T_max_measurement)

        f_grid, T_grid = np.meshgrid(grid_frequency, grid_temperature)
        eps_real, eps_imag = self.fit_real_and_imaginary_part_at_f_and_T(f_grid, T_grid)

        df_grid = pd.DataFrame(
            {
                "f": f_grid.ravel(),
                "T": T_grid.ravel(),
                "eps_real": np.ravel(eps_real),
                "eps_imag": np.ravel(eps_imag),
            }
        )
        return df_grid

    @staticmethod
    def plot_grid(df: pd.DataFrame,
                  save_path: str | Path,
                  no_levels: int = 100,
                  f_min: float | None = None, f_max: float | None = None,
                  T_min: float | None = None, T_max: float | None = None) -> None:
        """
        Plot |eps| and phase(eps) as contour maps vs. f (kHz) and T (°C) with shared color scales for magnitude and phase.

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
