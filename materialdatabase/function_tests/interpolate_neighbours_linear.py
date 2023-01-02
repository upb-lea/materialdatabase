from materialdatabase.material_data_base_functions import *
from materialdatabase.material_data_base_classes import *

mdb = MaterialDatabase()

list_of_permittivity_dicts = mdb.load_permittivity_measurement(material_name="N49", datasource="measurements", measurement_setup="LEA_LK")

T = 64
f = 450000

neighbourhood = create_permittivity_neighbourhood(T=T, f=f, list_of_permittivity_dicts=list_of_permittivity_dicts)

print(f"{neighbourhood = }")


print(interpolate_neighbours_linear(T=T, f=f, neighbours=neighbourhood))

