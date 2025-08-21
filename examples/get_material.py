"""Example file to show how to get different material data."""
import logging

import materialdatabase as mdb

# configure logging to show terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# init a material database instance
mdb_data = mdb.Data()
material_name = mdb.Material.N49

permeability = mdb_data.get_complex_permeability(material=material_name,
                                                 data_source=mdb.DataSource.TDK_MDT,
                                                 pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
print(f"Exemplary complex permeability data: \n {permeability.measurement_data} \n")


permittivity = mdb_data.get_complex_permittivity(material=material_name,
                                                 data_source=mdb.DataSource.LEA_MTB)
print(f"Exemplary complex permittivity data: \n {permittivity.measurement_data} \n ")
