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

# Flags
MU_ABS = True
PV = True


def fit_complex_permeability_tdk_mdt_example(mu_abs_flag: bool, pv_flag: bool,
                                             is_plot: bool = True) -> None:
    """Process data with the ComplexPermeability class example.

    :param mu_abs_flag: flag to show mu_abs
    :type mu_abs_flag: bool
    :param pv_flag: flat to show pv
    :type pv_flag: bool
    :param is_plot: True to show visual outputs
    :type is_plot: bool
    """
    # init a material database instance
    mdb_data = mdb.Data()

    # load ComplexPermeability instance
    permeability = mdb_data.get_complex_permeability(material=mdb.Material.N49,
                                                     data_source=mdb.DataSource.TDK_MDT,
                                                     pv_fit_function=mdb.FitFunction.Steinmetz)
    permeability.fit_permeability_magnitude()
    params_SE = permeability.fit_losses()

    # copy measurement in an extra dataframe
    df_mat = permeability.measurement_data.copy(deep=True)
    df_mat["mu_abs"] = np.sqrt(df_mat["mu_real"] ** 2 + df_mat["mu_imag"] ** 2)
    df_mat["pv"] = pv_mag(df_mat["f"].to_numpy(), -df_mat["mu_imag"].to_numpy() * mu_0, df_mat["b"].to_numpy() / df_mat["mu_abs"].to_numpy() / mu_0)

    if mu_abs_flag:
        df_mat["mu_abs_fitted"] = permeability.mu_a_fit_function.get_function()((df_mat["f"].to_numpy(),
                                                                                 df_mat["T"].to_numpy(),
                                                                                 df_mat["b"].to_numpy()),
                                                                                *permeability.params_mu_a)
        rel_error_mu_abs = abs(df_mat["mu_abs_fitted"] - df_mat["mu_abs"]) / df_mat["mu_abs"]
        print(f"MRE (mu_abs) = {np.mean(rel_error_mu_abs)}")

        # # plot measurement vs fitted data
        y_columns = ["mu_abs", "mu_abs_fitted"]
        styles_mu = {
            "mu_abs": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "Measured"}),
            "mu_abs_fitted": cast(StyleDict, {"marker": "*", "color": colors().compare1, "label": "Fitted"}),
        }
        if is_plot:
            plot_mu_all(df=df_mat[(df_mat["T"].isin([25, 40, 60, 80, 100, 120])) & (df_mat["f"].isin([25e3, 100e3, 500e3, 1e6]))],
                        y_columns=y_columns,
                        styles=styles_mu,
                        annotate=False)

    if pv_flag:
        df_mat["pv_fitted_SE"] = mdb.steinmetz_qT((df_mat["f"].to_numpy(), df_mat["T"].to_numpy(), df_mat["b"].to_numpy()), *params_SE)
        rel_error_SE = abs(df_mat["pv_fitted_SE"] - df_mat["pv"]) / df_mat["pv"]
        print(f"MRE (SE) = {np.mean(rel_error_SE)}")

        # fit the enhanced Steinmetz equation
        permeability.pv_fit_function = mdb.FitFunction.enhancedSteinmetz
        params_eSE = permeability.fit_losses()
        df_mat["pv_fitted_eSE"] = mdb.enhanced_steinmetz_qT((df_mat["f"].to_numpy(), df_mat["T"].to_numpy(), df_mat["b"].to_numpy()), *params_eSE)
        rel_error_eSE = abs(df_mat["pv_fitted_eSE"] - df_mat["pv"]) / df_mat["pv"]
        print(f"MRE (eSE) = {np.mean(rel_error_eSE)}")

        # plot the fitted data
        y_columns = ["pv", "pv_fitted_eSE", "pv_fitted_SE"]
        styles_losses: Dict[str, StyleDict] = {
            "pv": cast(StyleDict, {"marker": "x", "color": colors().gtruth, "label": "Measured"}),
            "pv_fitted_SE": cast(StyleDict, {"marker": ".", "color": colors().compare1, "label": "Fitted SE"}),
            "pv_fitted_eSE": cast(StyleDict, {"marker": "*", "color": colors().compare2, "label": "Fitted eSE"}),
        }
        if is_plot:
            plot_combined_loss(df=df_mat[(df_mat["T"].isin([25, 60, 100])) & (df_mat["f"].isin([25e3, 100e3, 500e3, 1e6]))],
                               y_columns=y_columns,
                               styles=styles_losses,
                               annotate=False)


if __name__ == '__main__':
    fit_complex_permeability_tdk_mdt_example(mu_abs_flag=MU_ABS, pv_flag=PV, is_plot=True)
