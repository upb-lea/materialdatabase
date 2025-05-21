"""Example file to show how to plot a overview of the existing material data."""
import materialdatabase as mdb
import logging

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def get_overview() -> None:
    """Init the materialdatabase and plot the material overview in a colored table. """

    mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)
    mdb_data.plot_available_data()


if __name__ == "__main__":
    get_overview()
