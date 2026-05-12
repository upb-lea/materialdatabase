"""Exemplary usage of get_datasheet_curve()."""

# python libraries
import logging

# 3rd party libraries
from matplotlib import pyplot as plt
import pandas as pd

# own libraries
import materialdatabase as mdb

# configure logging to show terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def get_datasheet_curve_example(is_plot: bool = False) -> None:
    """get_datasheet_curve() example.

    :param is_plot: True to show plots
    :type is_plot: bool
    """
    # init a material database instance
    mdb_data = mdb.Data()
    material_name = mdb.Material._3C95

    b_over_h_at_f_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.b_over_h_at_f_T)
    print(b_over_h_at_f_T.head())
    if is_plot:
        for temperature in pd.unique(b_over_h_at_f_T["T"]):
            for frequency in pd.unique(b_over_h_at_f_T["f"]):
                boundary = (b_over_h_at_f_T["T"] == temperature) & (b_over_h_at_f_T["f"] == frequency)
                plt.plot(b_over_h_at_f_T.loc[boundary]["h"], b_over_h_at_f_T.loc[boundary]["b"], 'o', label=f"{temperature} °C, {frequency} Hz")
        plt.xlabel("H / (A / m)")
        plt.ylabel("B / T")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    df_curve_mu_amplitude = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.mu_amplitude_over_b_at_T)
    print(df_curve_mu_amplitude.head())
    if is_plot:
        for temperature in pd.unique(df_curve_mu_amplitude["T"]):
            boundary = (df_curve_mu_amplitude["T"] == temperature)
            plt.plot(df_curve_mu_amplitude.loc[boundary]["b"], df_curve_mu_amplitude.loc[boundary]["mu"], 'o', label=f"{temperature} °C")
        plt.xlabel("B / T")
        plt.ylabel("mu_r")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    p_v_over_b_at_f_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_b_at_f_T)
    print(p_v_over_b_at_f_T.head())
    if is_plot:
        for temperature in pd.unique(p_v_over_b_at_f_T["T"]):
            for frequency in pd.unique(p_v_over_b_at_f_T["f"]):
                boundary = (p_v_over_b_at_f_T["T"] == temperature) & (p_v_over_b_at_f_T["f"] == frequency)
                plt.loglog(p_v_over_b_at_f_T.loc[boundary]["b"], p_v_over_b_at_f_T.loc[boundary]["p_v"], 'o', label=f"{temperature} °C, {frequency} Hz")
        plt.xlabel("B / T")
        plt.ylabel("P_v / W")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    p_v_over_f_at_b_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_f_at_b_T)
    print(p_v_over_f_at_b_T.head())
    if is_plot:
        for temperature in pd.unique(p_v_over_f_at_b_T["T"]):
            for flux_density in pd.unique(p_v_over_f_at_b_T["b"]):
                boundary = (p_v_over_f_at_b_T["T"] == temperature) & (p_v_over_f_at_b_T["b"] == flux_density)
                plt.loglog(p_v_over_f_at_b_T.loc[boundary]["f"], p_v_over_f_at_b_T.loc[boundary]["p_v"], 'o', label=f"{temperature} °C, {flux_density} T")
        plt.xlabel("f / Hz")
        plt.ylabel("P_v / W")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    p_v_over_T_at_f_b = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_T_at_f_b)
    print(p_v_over_T_at_f_b.head())
    if is_plot:
        for frequency in pd.unique(p_v_over_T_at_f_b["f"]):
            for flux_density in pd.unique(p_v_over_T_at_f_b["b"]):
                boundary = (p_v_over_T_at_f_b["b"] == flux_density) & (p_v_over_T_at_f_b["f"] == frequency)
                plt.semilogy(p_v_over_T_at_f_b.loc[boundary]["T"], p_v_over_T_at_f_b.loc[boundary]["p_v"], 'o', label=f"{frequency} Hz, {flux_density} T")
        plt.xlabel("T / °C")
        plt.ylabel("P_v / W")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    small_signal_mu_imag_over_f_at_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_imag_over_f_at_T)
    print(small_signal_mu_imag_over_f_at_T.head())
    if is_plot:
        plt.loglog(small_signal_mu_imag_over_f_at_T["f"], small_signal_mu_imag_over_f_at_T["mu_imag"], 'o', label="")
        plt.xlabel("f / Hz")
        plt.ylabel("mu_imag")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    small_signal_mu_initial_over_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_initial_over_T)
    print(small_signal_mu_initial_over_T.head())
    if is_plot:
        plt.plot(small_signal_mu_initial_over_T["T"], small_signal_mu_initial_over_T["mu"], 'o', label="")
        plt.xlabel("T / °C")
        plt.ylabel("mu_abs")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    small_signal_mu_real_over_f_at_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_real_over_f_at_T)
    print(small_signal_mu_real_over_f_at_T.head())
    if is_plot:
        plt.loglog(small_signal_mu_real_over_f_at_T["f"], small_signal_mu_real_over_f_at_T["mu_real"], 'o', label="")
        plt.xlabel("f / Hz")
        plt.ylabel("mu_real")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    get_datasheet_curve_example(True)
