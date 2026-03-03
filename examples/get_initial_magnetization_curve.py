
"""Example to get the initialization magnetization curve from data, if data are available."""

import logging


# Debug
import matplotlib.pyplot as plt

# own libraries
import materialdatabase as mdb

# configure logging to show terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def get_datasheet_curve_example():
    """get_datasheet_curve() example."""
    # init a material database instance
    mdb_data = mdb.Data()
    material_name = mdb.Material.N49

    # Initial magnetization curve at 25 degree
    df_init_mag_curve_25 = mdb_data.get_initial_magnetization_curve(material_name, 500, 25)
    # Initial magnetization curve at 100 degree
    df_init_mag_curve_100 = mdb_data.get_initial_magnetization_curve(material_name, 500, 100)
    # Interpolated initial magnetization curve at 40 degree
    df_init_mag_curve_40 = mdb_data.get_initial_magnetization_curve(material_name, 500, 40)

    # plot all curves
    plt.plot(df_init_mag_curve_25["h"], df_init_mag_curve_25["b"], 'o', label='at 25°')
    plt.plot(df_init_mag_curve_100["h"], df_init_mag_curve_100["b"], 'o', label='at 100°')
    plt.plot(df_init_mag_curve_40["h"], df_init_mag_curve_40["b"], 'o', label='at 40°')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    get_datasheet_curve_example()
