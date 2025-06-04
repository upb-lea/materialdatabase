"""Example to compare material data from different measurement setups or materials."""

import logging
from itertools import product
from typing import cast

import numpy as np
import pandas as pd

import materialdatabase as mdb
from materialdatabase import get_user_colors as colors
from materialdatabase.processing.plot import plot_combined_loss, plot_mu_all, StyleDict

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Flags
PLOT_MU_ABS = True
PLOT_PV = True

# Comparison settings
FREQS = np.linspace(3e5, 4e5, 3)
TEMPS = [100]
FLUX_DENSITIES = np.linspace(0.05, 0.15, 3)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)

materials = {
    "N49": mdb_data.get_complex_permeability(mdb.Material.N49, mdb.MeasurementSetup.TDK_MDT),
    "N95": mdb_data.get_complex_permeability(mdb.Material.N95, mdb.MeasurementSetup.TDK_MDT),
    "N87": mdb_data.get_complex_permeability(mdb.Material.N87, mdb.MeasurementSetup.TDK_MDT)
}

df_common = pd.DataFrame(
    product(FREQS, TEMPS, FLUX_DENSITIES),
    columns=["f", "T", "b"]
)

# ─────────────────────────────────────────────
# Plot Permeability Magnitude
# ─────────────────────────────────────────────

if PLOT_MU_ABS:
    for label, material in materials.items():
        params = material.fit_permeability_magnitude(mu_a_fit_function=mdb.FitFunction.mu_abs_fTb)
        df_common[f"mu_abs_{label}"] = mdb.fit_mu_abs_fTb(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()),
            *params
        )

    styles_mu = {
        "mu_abs_N49": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "N49"}),
        "mu_abs_N95": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "N95"}),
        "mu_abs_N87": cast(StyleDict, {"marker": "o", "color": colors().compare2, "label": "N87"})
    }

    plot_mu_all(df=df_common, y_columns=list(styles_mu.keys()), styles=styles_mu, annotate=False)

# ─────────────────────────────────────────────
# Plot Power Loss
# ─────────────────────────────────────────────

if PLOT_PV:
    for label, material in materials.items():
        params = material.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)
        df_common[f"pv_{label}"] = mdb.enhanced_steinmetz_qT(
            (df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()),
            *params
        )

    styles_pv = {
        "pv_N49": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "N49"}),
        "pv_N95": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "N95"}),
        "pv_N87": cast(StyleDict, {"marker": "o", "color": colors().compare2, "label": "N87"})
    }

    plot_combined_loss(df=df_common, y_columns=list(styles_pv.keys()), styles=styles_pv, annotate=False)
