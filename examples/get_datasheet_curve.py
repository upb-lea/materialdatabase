"""Exemplary usage of get_datasheet_curve()."""

# python libraries
import logging

# own libraries
import materialdatabase as mdb

# configure logging to show terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def get_datasheet_curve_example():
    """get_datasheet_curve() example."""
    # init a material database instance
    mdb_data = mdb.Data()
    material_name = mdb.Material.N49

    b_over_h_at_f_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.b_over_h_at_f_T)
    print(b_over_h_at_f_T.head())

    df_curve_mu_amplitude = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.mu_amplitude_over_b_at_T)
    print(df_curve_mu_amplitude.head())

    p_v_over_b_at_f_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_b_at_f_T)
    print(p_v_over_b_at_f_T.head())

    p_v_over_f_at_b_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_f_at_b_T)
    print(p_v_over_f_at_b_T.head())

    p_v_over_T_at_f_b = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.p_v_over_T_at_f_b)
    print(p_v_over_T_at_f_b.head())

    small_signal_mu_imag_over_f_at_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_imag_over_f_at_T)
    print(small_signal_mu_imag_over_f_at_T.head())

    small_signal_mu_initial_over_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_initial_over_T)
    print(small_signal_mu_initial_over_T.head())

    small_signal_mu_real_over_f_at_T = mdb_data.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_real_over_f_at_T)
    print(small_signal_mu_real_over_f_at_T.head())


if __name__ == "__main__":
    get_datasheet_curve_example()
