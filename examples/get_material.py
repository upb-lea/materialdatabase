"""Example file to show how to get different material data."""
import logging
import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# init a material database instance
mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)

# get exemplary complex material data as pandas dataframes:
df_mu = mdb_data.get_complex_data_set(material=mdb.Material.N49,
                                      measurement_setup=mdb.MeasurementSetup.TDK_MDT,
                                      data_type=mdb.ComplexDataType.complex_permeability)
print(f"Exemplary complex permeability data: \n {df_mu} \n")

df_eps = mdb_data.get_complex_data_set(material=mdb.Material.N49,
                                       measurement_setup=mdb.MeasurementSetup.LEA_MTB,
                                       data_type=mdb.ComplexDataType.complex_permittivity)
print(f"Exemplary complex permittivity data: \n {df_eps} \n ")
