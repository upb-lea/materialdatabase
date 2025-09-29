"""Example file to show how to get different material data."""
from pathlib import Path
import logging
import numpy as np

import materialdatabase as mdb
from materialdatabase.meta.config import get_user_paths

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def export_material2grid_example():
    """Get different material data example."""
    # init a material database instance
    mdb_data = mdb.Data()
    path2grid_export = Path(get_user_paths().grid_export_data)
    print(f"Data exported to {path2grid_export}")
    material_name = mdb.Material.N49

    permeability = mdb_data.get_complex_permeability(material=material_name,
                                                     data_source=mdb.DataSource.LEA_MTB,
                                                     pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
    permeability.export_to_txt(path2grid_export.joinpath(f"{material_name.value}_permeability_grid.txt"),
                               frequencies=np.linspace(1e5, 1.5e6, 30),
                               temperatures=np.linspace(25, 70, 10),
                               b_vals=np.linspace(0, 0.2, 10))
    print(f"Exemplary complex permeability data: \n {permeability.measurement_data} \n")

    permittivity = mdb_data.get_complex_permittivity(material=material_name,
                                                     data_source=mdb.DataSource.LEA_MTB)
    permittivity.export_to_txt(path2grid_export.joinpath(f"{material_name.value}_permittivity_grid.txt"),
                               frequencies=np.linspace(1e5, 1.5e6, 30),
                               temperatures=np.linspace(25, 70, 10))
    print(f"Exemplary complex permittivity data: \n {permittivity.measurement_data} \n ")


if __name__ == '__main__':
    export_material2grid_example()
