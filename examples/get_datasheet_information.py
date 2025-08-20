"""Example file to show how to get general information from the data sheet."""
import logging

import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# init a material database instance
mdb_data = mdb.Data()
material_name = mdb.Material.N49

resistivity = mdb_data.get_datasheet_information(material=material_name,
                                                 attribute=mdb.DatasheetAttribute.Resistivity)

print(resistivity)

mu_initial = mdb_data.get_datasheet_information(material=material_name,
                                                attribute=mdb.DatasheetAttribute.InitialPermeability)

print(mu_initial)

bsat = mdb_data.get_datasheet_information(material=material_name,
                                          attribute=mdb.DatasheetAttribute.SaturationFluxDensity100)

print(bsat)

density = mdb_data.get_datasheet_information(material=material_name,
                                             attribute=mdb.DatasheetAttribute.Density)

print(density)
