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
from materialdatabase.meta.data_classes import ComplexPermeabilityPlotConfig, ComplexPermeabilityConfig

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
materials_config: dict[str, ComplexPermeabilityPlotConfig] = {
    "N49": ComplexPermeabilityPlotConfig(
        mat_cfg=ComplexPermeabilityConfig(
            material=mdb.Material.N49,
            setup=mdb.MeasurementSetup.TDK_MDT,
            pv_fit_function=mdb.FitFunction.enhancedSteinmetz
        ),
        enabled=True,
        label="N49 (TDK)",
        color=colors().gtruth,
        marker="x"
    ),
    "N95": ComplexPermeabilityPlotConfig(
        mat_cfg=ComplexPermeabilityConfig(
            material=mdb.Material.N95,
            setup=mdb.MeasurementSetup.TDK_MDT,
            pv_fit_function=mdb.FitFunction.enhancedSteinmetz
        ),
        enabled=True,
        label="N95 (TDK)",
        color=colors().compare1,
        marker="*"
    ),
    "N87": ComplexPermeabilityPlotConfig(
        mat_cfg=ComplexPermeabilityConfig(
            material=mdb.Material.N87,
            setup=mdb.MeasurementSetup.TDK_MDT,
            pv_fit_function=mdb.FitFunction.enhancedSteinmetz
        ),
        enabled=True,
        label="N87 (TDK)",
        color=colors().compare2,
        marker="o"
    ),
    "PC200": ComplexPermeabilityPlotConfig(
        mat_cfg=ComplexPermeabilityConfig(
            material=mdb.Material.PC200,
            setup=mdb.MeasurementSetup.TDK_MDT,
            pv_fit_function=mdb.FitFunction.enhancedSteinmetz
        ),
        enabled=False,
        label="PC200 (TDK)",
        color=colors().compare3,
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

        logging.info(f"Computing permeability for: {cfg.label} (Setup: {cfg.mat_cfg.setup.name})")
        permeability = mdb_data.get_complex_permeability(material=cfg.mat_cfg.material,
                                                         measurement_setup=cfg.mat_cfg.setup,
                                                         pv_fit_function=cfg.mat_cfg.pv_fit_function)
        params = permeability.fit_permeability_magnitude()
        col = f"mu_abs_{key}"
        df_common[col] = permeability.mu_a_fit_function.get_function()(
            (df_common["f"].to_numpy(),
             df_common["T"].to_numpy(),
             df_common["b"].to_numpy()),
            *params
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

        logging.info(f"Computing permeability for: {cfg.label} (Setup: {cfg.mat_cfg.setup.name})")
        permeability = mdb_data.get_complex_permeability(material=cfg.mat_cfg.material,
                                                         measurement_setup=cfg.mat_cfg.setup,
                                                         pv_fit_function=cfg.mat_cfg.pv_fit_function)
        params = permeability.fit_losses()
        col = f"pv_{key}"
        df_common[col] = permeability.pv_fit_function.get_function()(
            (df_common["f"].to_numpy(),
             df_common["T"].to_numpy(),
             df_common["b"].to_numpy()),
            *params
        )
        styles_pv[col] = cast(StyleDict, {
            "marker": cfg.marker,
            "color": cfg.color,
            "label": cfg.label
        })

    plot_combined_loss(df=df_common, y_columns=list(styles_pv.keys()), styles=styles_pv, annotate=False)
