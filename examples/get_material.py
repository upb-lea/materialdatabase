"""Example file to show how to get different material data."""
import logging
import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# init a material database instance
mdb_data = mdb.Data()

mu_N49 = mdb_data.get_complex_permeability(material=mdb.Material.N49,
                                           measurement_setup=mdb.MeasurementSetup.TDK_MDT,
                                           mu_a_fit_function=mdb.FitFunction.mu_abs_TDK_MDT,
                                           pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
print(f"Exemplary complex permeability data: \n {mu_N49.measurement_data} \n")


eps_N49 = mdb_data.get_complex_permittivity(material=mdb.Material.N49,
                                            measurement_setup=mdb.MeasurementSetup.LEA_MTB)
print(f"Exemplary complex permittivity data: \n {eps_N49.measurement_data} \n ")
