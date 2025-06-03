"""Example file to show how to compare material data of different measurement setups or different materials."""
import pandas as pd
import materialdatabase as mdb
from materialdatabase import get_user_colors as colors
import logging
import numpy as np
from materialdatabase.processing.plot import plot_combined_loss, plot_mu_all, StyleDict
from typing import cast
from itertools import product

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Flags
MU_ABS = True
PV = True

# init a material database instance
mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)

# load ComplexMaterial instances
mu_N49_TDK_MDT = mdb_data.get_complex_permeability(material=mdb.Material.N49, measurement_setup=mdb.MeasurementSetup.TDK_MDT)
mu_N49_LEA_MTB = mdb_data.get_complex_permeability(material=mdb.Material.N49, measurement_setup=mdb.MeasurementSetup.LEA_MTB)

# copy TDK-MDT data in an extra dataframe
df_TDK = mu_N49_TDK_MDT.measurement_data.copy(deep=True)

# copy TDK-MDT data in an extra dataframe
df_LEA = mu_N49_LEA_MTB.measurement_data.copy(deep=True)

# create a "common" dataframe, which is used for the comparison
# df_common = mu_N49_LEA_MTB.measurement_data.copy(deep=True).drop(columns=['mu_imag', 'mu_real'])  # simple copy of LEA-MTB measurements
f_ = np.linspace(1e5, 1e6, 5)
T_ = [25]
b_ = np.linspace(0.025, 0.1, 5)
# Get all combinations
combinations = list(product(f_, T_, b_))
# Create DataFrame
df_common = pd.DataFrame(combinations, columns=["f", "T", "b"])

if MU_ABS:
    # Fitting of TDK-MDT data
    params_mu_abs_TDK = mu_N49_TDK_MDT.fit_permeability_magnitude()
    df_common["mu_abs_TDK"] = mdb.fit_mu_abs_fTb((df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params_mu_abs_TDK)

    # Fitting of LEA-MTB data
    params_mu_abs_LEA = mu_N49_LEA_MTB.fit_permeability_magnitude()
    df_common["mu_abs_LEA"] = mdb.fit_mu_abs_fTb((df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params_mu_abs_LEA)

    # plot TDK vs LEA data
    y_columns = ["mu_abs_TDK", "mu_abs_LEA"]
    styles_mu = {
        "mu_abs_TDK": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "TDK"}),
        "mu_abs_LEA": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "LEA"}),
    }
    plot_mu_all(df=df_common,
                y_columns=y_columns,
                styles=styles_mu,
                annotate=False)

if PV:
    # fit the Steinmetz equation
    params_pv_TDK = mu_N49_TDK_MDT.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)
    df_common["pv_TDK"] = mdb.enhanced_steinmetz_qT((df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params_pv_TDK)

    # fit the Steinmetz equation
    params_pv_LEA = mu_N49_LEA_MTB.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)
    df_common["pv_LEA"] = mdb.enhanced_steinmetz_qT((df_common["f"].to_numpy(), df_common["T"].to_numpy(), df_common["b"].to_numpy()), *params_pv_LEA)

    # plot TDK vs LEA data
    y_columns = ["pv_TDK", "pv_LEA"]
    styles_pv = {
        "pv_TDK": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "TDK"}),
        "pv_LEA": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "LEA"}),
    }
    plot_combined_loss(df=df_common,
                       y_columns=y_columns,
                       styles=styles_pv,
                       annotate=False)
