"""Compare material data from different measurement setups.

This script compares core loss and permeability magnitude for N49 material
measured using TDK-MDT and LEA-MTB setups. It fits both properties and plots
them across a defined sweep of frequency, temperature, and flux density.
"""

import logging
from itertools import product
from typing import cast

import numpy as np
import pandas as pd

import materialdatabase as mdb
from materialdatabase import get_user_colors as colors
from materialdatabase.processing.plot import plot_combined_loss, plot_mu_all, StyleDict
from materialdatabase.meta.data_classes import MaterialPlotConfig

# ---------------------------------------------
# Configuration
# ---------------------------------------------

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Flags to control which plots to generate
PLOT_MU_ABS = True
PLOT_PV = True

# Operating points of interest
FREQS = np.linspace(1e5, 1e6, 5)               # Frequency in Hz
TEMPS = [25]                                   # Temperature in °C
FLUX_DENSITIES = np.linspace(0.025, 0.1, 5)    # Flux density in T

# Materials to compare
materials_config: dict[str, MaterialPlotConfig] = {
    "N49_TDK": MaterialPlotConfig(
        enabled=True,
        material=mdb.Material.N49,
        setup=mdb.MeasurementSetup.TDK_MDT,
        label="N49 (TDK-MDT)",
        color=colors().gtruth,
        marker="x"
    ),
    "N49_LEA": MaterialPlotConfig(
        enabled=True,
        material=mdb.Material.N49,
        setup=mdb.MeasurementSetup.LEA_MTB,
        label="N49 (LEA-MTB)",
        color=colors().compare1,
        marker="*"
    )
}

# ---------------------------------------------
# Load Material Data and Prepare Grid
# ---------------------------------------------

mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)

# Create sweep grid of all combinations
df_common = pd.DataFrame(
    product(FREQS, TEMPS, FLUX_DENSITIES),
    columns=["f", "T", "b"]
)

# ---------------------------------------------
# Plot Permeability Magnitude
# ---------------------------------------------

if PLOT_MU_ABS:
    styles_mu = {}
    for key, cfg in materials_config.items():
        if not cfg.enabled:
            continue

        logging.info(f"Fitting permeability |μ| for: {cfg.label} ({cfg.setup.name})")
        material = mdb_data.get_complex_permeability(cfg.material, cfg.setup)
        params = material.fit_permeability_magnitude(mdb.FitFunction.mu_abs_fTb)

        col_name = f"mu_abs_{key}"
        df_common[col_name] = mdb.fit_mu_abs_fTb(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()),
            *params
        )

        styles_mu[col_name] = cast(StyleDict, {
            "marker": cfg.marker,
            "color": cfg.color,
            "label": cfg.label
        })

    plot_mu_all(df=df_common, y_columns=list(styles_mu.keys()), styles=styles_mu, annotate=False)

# ---------------------------------------------
# Plot Core Loss
# ---------------------------------------------

if PLOT_PV:
    styles_pv = {}
    for key, cfg in materials_config.items():
        if not cfg.enabled:
            continue

        logging.info(f"Fitting power loss for: {cfg.label} ({cfg.setup.name})")
        material = mdb_data.get_complex_permeability(cfg.material, cfg.setup)
        params = material.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)

        col_name = f"pv_{key}"
        df_common[col_name] = mdb.enhanced_steinmetz_qT(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()),
            *params
        )

        styles_pv[col_name] = cast(StyleDict, {
            "marker": cfg.marker,
            "color": cfg.color,
            "label": cfg.label
        })

    plot_combined_loss(df=df_common, y_columns=list(styles_pv.keys()), styles=styles_pv, annotate=False)
