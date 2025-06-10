"""Example file to show how to process data with the ComplexPermeability class."""
import materialdatabase as mdb
from materialdatabase import get_user_colors as colors
import logging
import numpy as np
from materialdatabase.processing.utils.physic import pv_mag
from materialdatabase.processing.utils.constants import mu_0
from materialdatabase.processing.plot import plot_combined_loss, plot_mu_all, StyleDict
from typing import cast, Dict

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)

# Flags
MU_ABS = True
PV = True

# init a material database instance
mdb_data = mdb.Data()

# load ComplexMaterial instance
mu_N49 = mdb_data.get_complex_permeability(material=mdb.Material.N49, measurement_setup=mdb.MeasurementSetup.TDK_MDT)

# copy measurement in an extra dataframe
df_N49 = mu_N49.measurement_data.copy(deep=True)
df_N49["mu_abs"] = np.sqrt(df_N49["mu_real"] ** 2 + df_N49["mu_imag"] ** 2)
df_N49["pv"] = pv_mag(df_N49["f"].to_numpy(), -df_N49["mu_imag"].to_numpy() * mu_0, df_N49["b"].to_numpy() / df_N49["mu_abs"].to_numpy() / mu_0)

if MU_ABS:
    # Fitting of the permeability magnitude mu_abs
    params_mu_abs = mu_N49.fit_permeability_magnitude(mdb.FitFunction.mu_abs_fTb)
    df_N49["mu_abs_fitted"] = mdb.fit_mu_abs_fTb((df_N49["f"].to_numpy(), df_N49["T"].to_numpy(), df_N49["b"].to_numpy()), *params_mu_abs)
    rel_error_mu_abs = abs(df_N49["mu_abs_fitted"] - df_N49["mu_abs"]) / df_N49["mu_abs"]
    print(f"MRE (mu_abs) = {np.mean(rel_error_mu_abs)}")

    # # plot measurement vs fitted data
    y_columns = ["mu_abs", "mu_abs_fitted"]
    styles_mu = {
        "mu_abs": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "Measured"}),
        "mu_abs_fitted": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "Fitted"}),
    }
    plot_mu_all(df=df_N49[(df_N49["T"].isin([25, 60, 100])) & (df_N49["f"].isin([25e3, 100e3, 500e3, 1e6]))],
                y_columns=y_columns,
                styles=styles_mu,
                annotate=False)

if PV:
    # fit the Steinmetz equation
    params_SE = mu_N49.fit_losses(loss_fit_function=mdb.FitFunction.Steinmetz)
    df_N49["pv_fitted_SE"] = mdb.steinmetz_qT((df_N49["f"].to_numpy(), df_N49["T"].to_numpy(), df_N49["b"].to_numpy()), *params_SE)
    rel_error_SE = abs(df_N49["pv_fitted_SE"] - df_N49["pv"]) / df_N49["pv"]
    print(f"MRE (SE) = {np.mean(rel_error_SE)}")

    # fit the enhanced Steinmetz equation
    params_eSE = mu_N49.fit_losses(loss_fit_function=mdb.FitFunction.enhancedSteinmetz)
    df_N49["pv_fitted_eSE"] = mdb.enhanced_steinmetz_qT((df_N49["f"].to_numpy(), df_N49["T"].to_numpy(), df_N49["b"].to_numpy()), *params_eSE)
    rel_error_eSE = abs(df_N49["pv_fitted_eSE"] - df_N49["pv"]) / df_N49["pv"]
    print(f"MRE (eSE) = {np.mean(rel_error_eSE)}")

    # plot the fitted data
    y_columns = ["pv", "pv_fitted_eSE", "pv_fitted_SE"]
    styles_losses: Dict[str, StyleDict] = {
        "pv": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "Measured"}),
        "pv_fitted_SE": cast(StyleDict, {"marker": ".", "color": colors().compare1, "label": "Fitted SE"}),
        "pv_fitted_eSE": cast(StyleDict, {"marker": "*", "color": colors().compare2, "label": "Fitted eSE"}),
    }
    plot_combined_loss(df=df_N49[(df_N49["T"].isin([25, 60, 100])) & (df_N49["f"].isin([25e3, 100e3, 500e3, 1e6]))],
                       y_columns=y_columns,
                       styles=styles_losses,
                       annotate=False)
