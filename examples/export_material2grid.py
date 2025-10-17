"""Example file to show how to get different material data."""
from pathlib import Path
import logging
import numpy as np

import materialdatabase as mdb
from materialdatabase.meta.config import get_user_paths

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def export_material2grid_example(plot: bool, save_to_txt: bool) -> None:
    """
    Get different material data example.

    :param plot: whether to plot or not
    :param save_to_txt: whether to save to txt file
    """
    # init a material database instance
    mdb_data = mdb.Data()
    path2grid_export = Path(get_user_paths().grid_export_data)
    path2grid_plot = Path(get_user_paths().graphics)
    material_name = mdb.Material.N49

    # Permeability
    permeability_data_source = mdb.DataSource.LEA_MTB
    link2permeability_grid = path2grid_export.joinpath(f"{material_name.value}_{permeability_data_source.value}_permeability_grid.txt")
    link2permeability_plot = path2grid_plot.joinpath(f"{material_name.value}_{permeability_data_source.value}_permeability_grid.pdf")
    permeability = mdb_data.get_complex_permeability(material=material_name,
                                                     data_source=permeability_data_source,
                                                     pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
    df_permeability_grid = permeability.to_grid(grid_frequency=np.linspace(1e5, 1.5e6, 50),
                                                grid_temperature=np.linspace(25, 70, 20),
                                                grid_flux_density=np.linspace(0, 0.2, 50),
                                                f_min_measurement=1e5, f_max_measurement=None,
                                                T_min_measurement=28, T_max_measurement=None,
                                                b_min_measurement=None, b_max_measurement=0.15)

    # Permittivity
    permittivity_data_source = mdb.DataSource.LEA_MTB
    link2permittivity_grid = path2grid_export.joinpath(f"{material_name.value}_{permittivity_data_source.value}_permittivity_grid.txt")
    link2permittivity_plot = path2grid_plot.joinpath(f"{material_name.value}_{permittivity_data_source.value}_permittivity_grid.pdf")
    permittivity = mdb_data.get_complex_permittivity(material=material_name,
                                                     data_source=permittivity_data_source)
    df_permittivity_grid = permittivity.to_grid(grid_frequency=np.linspace(1e5, 1.5e6, 50),
                                                grid_temperature=np.linspace(25, 70, 20))

    if plot:
        permeability.plot_grid(df_permeability_grid,
                               save_path=link2permeability_plot,
                               temps=[25],
                               no_levels=20,
                               f_min=0.95e5, f_max=1.05e6,
                               b_min=20e-3, b_max=105e-3)
        permittivity.plot_grid(df_permittivity_grid,
                               no_levels=20,
                               save_path=link2permittivity_plot,
                               f_min=0.95e5, f_max=1.05e6)

    if save_to_txt:
        permeability.grid2txt(df_permeability_grid, link2permeability_grid)
        permittivity.grid2txt(df_permittivity_grid, link2permittivity_grid)
        print(f"Data exported to {path2grid_export}")


if __name__ == '__main__':
    export_material2grid_example(plot=True, save_to_txt=True)
