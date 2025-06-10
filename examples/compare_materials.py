"""Example to compare material data from different measurement setups or materials.

This script enables users to selectively visualize permeability and power loss
for various materials under customizable frequency, temperature, and flux conditions.
Each material can have its own measurement setup.
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
FREQS = np.linspace(5e5, 1e6, 5)         # Frequency range in Hertz
TEMPS = [100]                            # Temperatures in Celsius
FLUX_DENSITIES = np.linspace(0.04, 0.1, 10)  # Flux densities in Tesla

# Materials to evaluate
materials_config: dict[str, MaterialPlotConfig] = {
    "N49": MaterialPlotConfig(
        enabled=True,
        material=mdb.Material.N49,
        setup=mdb.MeasurementSetup.TDK_MDT,
        label="N49 (TDK)",
        color=colors().gtruth,
        marker="x"
    ),
    "N95": MaterialPlotConfig(
        enabled=False,
        material=mdb.Material.N95,
        setup=mdb.MeasurementSetup.TDK_MDT,
        label="N95 (TDK)",
        color=colors().compare1,
        marker="*"
    ),
    "N87": MaterialPlotConfig(
        enabled=False,
        material=mdb.Material.N87,
        setup=mdb.MeasurementSetup.TDK_MDT,
        label="N87 (TDK)",
        color=colors().compare2,
        marker="o"
    ),
    "PC200": MaterialPlotConfig(
        enabled=True,
        material=mdb.Material.PC200,
        setup=mdb.MeasurementSetup.TDK_MDT,
        label="PC200 (TDK)",
        color=colors().compare2,
        marker="v"
    )
}

# ---------------------------------------------
# Load Material Data
# ---------------------------------------------

mdb_data = mdb.Data()

# Create sweep grid
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

        logging.info(f"Computing permeability for: {cfg.label} (Setup: {cfg.setup.name})")
        material = mdb_data.get_complex_permeability(cfg.material, cfg.setup)
        params = material.fit_permeability_magnitude(mu_a_fit_function=mdb.FitFunction.mu_abs_fTb)
        col = f"mu_abs_{key}"
        df_common[col] = mdb.fit_mu_abs_fTb(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params
        )
        styles_mu[col] = cast(StyleDict, {
            "marker": cfg.marker,
            "color": cfg.color,
            "label": cfg.label
        })

    plot_mu_all(df=df_common, y_columns=list(styles_mu.keys()), styles=styles_mu, annotate=False)

# ---------------------------------------------
# Plot Power Loss
# ---------------------------------------------

if PLOT_PV:
    styles_pv = {}
    for key, cfg in materials_config.items():
        if not cfg.enabled:
            continue

        logging.info(f"Computing power loss for: {cfg.label} (Setup: {cfg.setup.name})")
        material = mdb_data.get_complex_permeability(cfg.material, cfg.setup)
        params = material.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)
        col = f"pv_{key}"
        df_common[col] = mdb.enhanced_steinmetz_qT(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params
        )
        styles_pv[col] = cast(StyleDict, {
            "marker": cfg.marker,
            "color": cfg.color,
            "label": cfg.label
        })

    plot_combined_loss(df=df_common, y_columns=list(styles_pv.keys()), styles=styles_pv, annotate=False)
