"""Functions to load material data from the TDK Magnetic Design Tool in the material database."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import materialdatabase as mdb
from materialdatabase.meta.data_enums import Material, FitFunction
from materialdatabase.processing.utils.physic import mu_imag_from_pv
from materialdatabase.processing.utils.constants import mu_0
from materialdatabase import get_user_colors as colors

logger = logging.getLogger(__name__)


def read_mu_abs_TDK_MDT(material_path: Path,
                        material_name: str,
                        temperatures: list[int],
                        b_lim: float = 0.3) -> pd.DataFrame:
    """
    Read mu_a vs. B data from TDK MDT xlsx files.

    :param material_path: Path to material-specific folder
    :param material_name: Name of material (string)
    :param temperatures: List of temperatures in °C for which mu_a data is available
    :param b_lim: maximum flux density (crop higher flux densities)
    :return: combined DataFrame of mu_a data
    """
    df_list = []

    for temp in temperatures:
        file = material_path / f"mu_a_{temp}_C_{material_name}.xlsx"
        df = pd.read_excel(file)
        df.columns = pd.Index(["b", "mu_a"])
        df["T"] = temp
        df["b"] = df["b"] / 1000  # Convert mT to T
        df = df[df["b"] <= b_lim]  # crop for b_lim
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def tdkmdt2pandas(
    data_dir: Path | str,
    material: Material,
    f_: list[int],
    T_: list[int],
    T_mu_a: list[int] | None = None,
    save2file: bool = True,
    pv_max: float | None = None,
) -> None:
    """
    Read and process TDK MDT data to generate complex permeability values.

    :param data_dir: Path to base data directory
    :param material: Material enum (e.g., mdb.Material.N49)
    :param f_: List of frequencies in kHz
    :param T_: List of temperatures in °C
    :param T_mu_a: List of temperatures for mu_a fitting
    :param save2file: Whether to save results to material DB
    :param pv_max: Optional max loss density (W/m³)
    """
    if T_mu_a is None:
        T_mu_a = [25, 100]

    data_dir = Path(data_dir)
    material_path = data_dir / material.name

    # Read mu_a data
    df_mu_a = read_mu_abs_TDK_MDT(material_path, material.name, T_mu_a)

    # fit permeability magnitude
    mu_a_fit_function = FitFunction.mu_abs_TDK_MDT.get_function()
    params_mu_abs, _ = curve_fit(
        mu_a_fit_function,
        (np.ones_like(df_mu_a["T"]), df_mu_a["T"], df_mu_a["b"]),
        df_mu_a["mu_a"],
        maxfev=100_000,
    )

    color_palette = [colors().compare1, colors().compare2, colors().compare3]
    # Check the quality of the plot
    for i, (T_val, group) in enumerate(df_mu_a.groupby("T")):
        plt.plot(group["b"], group["mu_a"], color=color_palette[i], label=f"T = {T_val} °C")
        mu_a_fit = mu_a_fit_function((np.ones_like(group["b"]), T_val * np.ones_like(group["b"]), group["b"]), *params_mu_abs)
        plt.plot(group["b"], mu_a_fit, "x", color=color_palette[i])

    plt.xlabel("B [T]")
    plt.ylabel("μ_a")
    plt.title("Permeability μ_a vs. B for Different Temperatures")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Collect loss density data
    data = []
    for frequency in f_:
        filename = material_path / f"pv_{frequency}_kHz_{material.name}.xlsx"
        df_pv_T = pd.read_excel(filename)

        for j, temperature in enumerate(T_):
            b = df_pv_T[f"Series{j + 1}.X"].to_numpy(dtype=float) / 1000  # mT → T
            pv = df_pv_T[f"Series{j + 1}.Y"].to_numpy(dtype=float) * 1000  # mW/cm³ → W/m³

            for bi, pvi in zip(b, pv, strict=False):
                if pv_max is None or pvi <= pv_max:
                    data.append({
                        "f": frequency * 1000,  # kHz → Hz
                        "T": temperature,
                        "b": bi,
                        "pv": pvi,
                    })
                else:
                    logger.info(
                        f"Exceeded max loss: T={temperature}°C, "
                        f"f={frequency}kHz, b={bi:.3f}T ({bi * 1000:.1f}mT)"
                    )

    df = pd.DataFrame(data).sort_values(by=["f", "T", "b"]).reset_index(drop=True)

    # Compute magnetic parameters
    df["mu_a"] = mu_a_fit_function((df["f"].to_numpy(), df["T"].to_numpy(), df["b"].to_numpy()), *params_mu_abs)
    H = df["b"] / df["mu_a"] / mu_0
    df["mu_imag"] = -mu_imag_from_pv(df["f"].to_numpy(), H.to_numpy(), df["pv"].to_numpy()) / mu_0
    df["mu_real"] = np.sqrt(np.maximum(df["mu_a"]**2 - df["mu_imag"]**2, 0))

    # Select final columns
    df = df[["f", "T", "b", "mu_real", "mu_imag"]]

    if save2file:
        mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)
        mdb_data.set_complex_permeability(
            material=material,
            data_source=mdb.DataSource.TDK_MDT,
            df=df,
        )
