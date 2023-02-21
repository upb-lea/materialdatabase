from materialdatabase.material_data_base_functions import *
from materialdatabase.material_data_base_classes import *

mdb = MaterialDatabase()

list_of_permittivity_dicts = mdb.load_permittivity_measurement(material_name="N49", datasource="measurements", measurement_setup="LEA_LK")

print(list_of_permittivity_dicts)

print(create_permittivity_neighbourhood(temperature=60, frequency=1e5, list_of_permittivity_dicts=list_of_permittivity_dicts))

