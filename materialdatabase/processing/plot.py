"""Plot functions."""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from typing import Dict, List, Union, TypedDict, Tuple


class StyleDict(TypedDict, total=False):
    """
    Style dictionary for configuring the appearance of a plot line.

    :param marker: Marker style for data points (e.g., 'o', '^', 's')
    :param color: Color of the line and markers (e.g., 'blue', '#ff5733')
    :param label: Label used in the legend
    """

    marker: str
    color: str
    label: str


def _flatten_y_columns(y_columns: Union[List[str], tuple]) -> List[str]:
    if isinstance(y_columns, tuple) and len(y_columns) == 1 and isinstance(y_columns[0], list):
        return y_columns[0]
    return list(y_columns)


def plot_loss_vs_temperature(ax: Axes, df: pd.DataFrame,
                             y_columns: Union[List[str], tuple], styles: Dict[str, StyleDict],
                             annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot core loss density versus temperature for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame with columns ['T', 'b', 'f'] plus loss columns
    :param y_columns: List or tuple of column names for y-axis data (loss columns)
    :param styles: Dictionary mapping y_column to dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (b, f) values
    :param y_label: If True, plot the y_label
    """
    y_columns = _flatten_y_columns(y_columns)

    for y_col in y_columns:
        style = styles[y_col]

        ax.semilogy(df["T"], df[y_col] / 1000,
                    style["marker"],
                    color=style["color"],
                    label=style["label"])

        for (b, f), group in df.groupby(["b", "f"]):
            sorted_group = group.sort_values("T")
            T_vals = sorted_group["T"]
            y_vals = sorted_group[y_col] / 1000

            ax.semilogy(T_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate:
                ax.text(T_vals.iloc[-1], y_vals.iloc[-1],
                        f"{int(b * 1000)} mT, {int(f / 1000)} kHz",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$T$ in $\mathrm{\degree C}$")
    if y_label:
        ax.set_ylabel(r"$p_\mathrm{v}$ in $\mathrm{kW}/\mathrm{m}^3$")
    ax.set_title("Loss vs. Temperature")
    ax.grid(True, which="both")
    ax.legend()


def plot_loss_vs_frequency(ax: Axes, df: pd.DataFrame,
                           y_columns: Union[List[str], tuple], styles: Dict[str, StyleDict],
                           annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot core loss density versus frequency for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame with columns ['f', 'b', 'T'] plus loss columns
    :param y_columns: List or tuple of column names for y-axis data (loss columns)
    :param styles: Dictionary mapping y_column to dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (b, T) values
    :param y_label: If True, plot the y_label
    """
    y_columns = _flatten_y_columns(y_columns)

    f_kHz = df["f"] / 1000

    for y_col in y_columns:
        style = styles[y_col]
        ax.loglog(f_kHz, df[y_col] / 1000,
                  style["marker"],
                  color=style["color"],
                  label=style["label"])

        for (b, T), group in df.groupby(["b", "T"]):
            sorted_group = group.sort_values("f")
            f_vals = sorted_group["f"] / 1000
            y_vals = sorted_group[y_col] / 1000

            ax.loglog(f_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate:
                ax.text(f_vals.iloc[-1], y_vals.iloc[-1],
                        f" {int(b * 1000)} mT, {int(T)} C",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$f$ in $\mathrm{kHz}$")
    if y_label:
        ax.set_ylabel(r"$p_\mathrm{v}$ in $\mathrm{kW}/\mathrm{m}^3$")
    ax.set_title("Loss vs. Frequency")
    ax.grid(True, which="both")
    ax.legend()


def plot_loss_vs_flux_density(ax: Axes, df: pd.DataFrame,
                              y_columns: Union[List[str], tuple], styles: Dict[str, StyleDict],
                              annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot core loss density versus flux density for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame with columns ['b', 'T', 'f'] plus loss columns
    :param y_columns: List or tuple of column names for y-axis data (loss columns)
    :param styles: Dictionary mapping y_column to dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (T, f) values
    :param y_label: If True, plot the y_label
    """
    y_columns = _flatten_y_columns(y_columns)

    for y_col in y_columns:
        style = styles[y_col]
        ax.loglog(df["b"] * 1000, df[y_col] / 1000,
                  style["marker"],
                  color=style["color"],
                  label=style["label"])

        for (T, f), group in df.groupby(["T", "f"]):
            sorted_group = group.sort_values("b")
            b_vals = sorted_group["b"] * 1000
            y_vals = sorted_group[y_col] / 1000

            ax.loglog(b_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate:
                ax.text(b_vals.iloc[-1], y_vals.iloc[-1],
                        f"{int(T)} C, {int(f / 1000)} kHz",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$b_\mathrm{peak}$ in $\mathrm{mT}$")
    if y_label:
        ax.set_ylabel(r"$p_\mathrm{v}$ in $\mathrm{kW}/\mathrm{m}^3$")
    ax.set_title("Loss vs. Flux Density")
    ax.grid(True, which="both")
    ax.legend()


def plot_combined_loss(df: pd.DataFrame,
                       y_columns: Union[List[str], Tuple[str, ...]],
                       styles: Dict[str, StyleDict],
                       annotate: bool = False) -> None:
    """
    Plot core loss density versus temperature, frequency, and flux density in subplots.

    :param df: DataFrame with columns including ['T', 'f', 'b'] and loss values
    :param y_columns: List or tuple of column names representing loss data series
    :param styles: Dictionary mapping each y_column to a style dict with keys like 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (b, f) values
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    plot_loss_vs_temperature(ax=axs[0], df=df, y_columns=y_columns, styles=styles, annotate=annotate, y_label=True)
    plot_loss_vs_frequency(ax=axs[1], df=df, y_columns=y_columns, styles=styles, annotate=annotate)
    plot_loss_vs_flux_density(ax=axs[2], df=df, y_columns=y_columns, styles=styles, annotate=annotate)
    plt.tight_layout()
    plt.show()


def plot_mu_vs_frequency(ax: Axes,
                         df: pd.DataFrame,
                         y_columns: Union[List[str], Tuple[str, ...]],
                         styles: Dict[str, StyleDict],
                         annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot magnetic permeability versus frequency for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame containing 'f', 'b', 'T', and permeability columns
    :param y_columns: List or tuple of permeability column names to plot
    :param styles: Dictionary mapping y_column to style dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (b, T) values
    :param y_label: If True, plot the y_label
    """
    for y_col in y_columns:
        style = styles[y_col]
        ax.plot(df["f"], df[y_col],
                style["marker"],
                color=style["color"],
                label=style["label"])

        for (b, T), group in df.groupby(["b", "T"]):
            sorted_group = group.sort_values("f")
            f_vals = sorted_group["f"]
            y_vals = sorted_group[y_col]
            ax.plot(f_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate and not f_vals.empty:
                ax.text(f_vals.iloc[-1], y_vals.iloc[-1],
                        f"{int(b * 1000)} mT, {int(T)} °C",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$f$ in $\mathrm{Hz}$")
    if y_label:
        ax.set_ylabel(r"$\mu$ (abs)")
    ax.set_title("Permeability vs. Frequency")
    ax.grid(True)
    ax.legend()


def plot_mu_vs_flux_density(ax: Axes,
                            df: pd.DataFrame,
                            y_columns: Union[List[str], Tuple[str, ...]],
                            styles: Dict[str, StyleDict],
                            annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot magnetic permeability versus flux density for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame containing 'b', 'f', 'T', and permeability columns
    :param y_columns: List or tuple of permeability column names to plot
    :param styles: Dictionary mapping y_column to style dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (T, f) values
    :param y_label: If True, plot the y_label
    """
    for y_col in y_columns:
        style = styles[y_col]
        ax.plot(df["b"], df[y_col],
                style["marker"],
                color=style["color"],
                label=style["label"])

        for (T, f), group in df.groupby(["T", "f"]):
            sorted_group = group.sort_values("b")
            b_vals = sorted_group["b"]
            y_vals = sorted_group[y_col]
            ax.plot(b_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate and not b_vals.empty:
                ax.text(b_vals.iloc[-1], y_vals.iloc[-1],
                        f"{int(T)} °C, {int(f / 1000)} kHz",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$b_\mathrm{peak}$ in $\mathrm{T}$")
    if y_label:
        ax.set_ylabel(r"$\mu$ (abs)")
    ax.set_title("Permeability vs. Flux Density")
    ax.grid(True)
    ax.legend()


def plot_mu_vs_temperature(ax: Axes,
                           df: pd.DataFrame,
                           y_columns: Union[List[str], Tuple[str, ...]],
                           styles: Dict[str, StyleDict],
                           annotate: bool = False, y_label: bool = False) -> None:
    """
    Plot magnetic permeability versus temperature for multiple data series.

    :param ax: Matplotlib Axes object to plot on
    :param df: DataFrame containing 'T', 'f', 'b', and permeability columns
    :param y_columns: List or tuple of permeability column names to plot
    :param styles: Dictionary mapping y_column to style dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with (b, f) values
    :param y_label: If True, plot the y_label
    """
    for y_col in y_columns:
        style = styles[y_col]
        ax.plot(df["T"], df[y_col],
                style["marker"],
                color=style["color"],
                label=style["label"])

        for (b, f), group in df.groupby(["b", "f"]):
            sorted_group = group.sort_values("T")
            T_vals = sorted_group["T"]
            y_vals = sorted_group[y_col]
            ax.plot(T_vals, y_vals, "-", color=style["color"], alpha=0.8)

            if annotate and not T_vals.empty:
                ax.text(T_vals.iloc[-1], y_vals.iloc[-1],
                        f"{int(b * 1000)} mT, {int(f / 1000)} kHz",
                        fontsize=8, color=style["color"],
                        ha="left", va="bottom", alpha=0.9)

    ax.set_xlabel(r"$T$ in $\mathrm{\degree C}$")
    if y_label:
        ax.set_ylabel(r"$\mu$ (abs)")
    ax.set_title("Permeability vs. Temperature")
    ax.grid(True)
    ax.legend()


def plot_mu_all(df: pd.DataFrame,
                y_columns: Union[List[str], Tuple[str, ...]],
                styles: Dict[str, StyleDict],
                annotate: bool = False) -> None:
    """
    Plot magnetic permeability versus frequency, flux density, and temperature for multiple data series.

    :param df: DataFrame with columns ['f', 'b', 'T'] and permeability values
    :param y_columns: List or tuple of column names representing permeability data series
    :param styles: Dictionary mapping each y_column to a style dict with keys 'marker', 'color', 'label'
    :param annotate: If True, annotate final data points with relevant parameter values
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    plot_mu_vs_temperature(ax=axs[0], df=df, y_columns=y_columns, styles=styles, annotate=annotate, y_label=True)
    plot_mu_vs_frequency(ax=axs[1], df=df, y_columns=y_columns, styles=styles, annotate=annotate)
    plot_mu_vs_flux_density(ax=axs[2], df=df, y_columns=y_columns, styles=styles, annotate=annotate)

    plt.tight_layout()
    plt.show()
