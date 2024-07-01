"""Script to write permittivity data by LEA_LK into the material database."""
import numpy as np

from materialdatabase.material_data_base_functions import *
from materialdatabase.material_data_base_classes import *


location = "C:/Users/tpiepe/sciebo/Exchange_FEMMT/05_Materials/data/2022_10_10_Ferrite_mu_eps_Data_Keuck/"
# filename = create_file_name_LEA_LK()
# print(filename)

mdb = MaterialDatabase()

# N49 - LEA_LK (30, 60, 80, 100) x (100000, 200000, 300000, 400000, 500000)
# create_permittivity_measurement_in_database("N49", "LEA_LK", company="Paderborn University", date="2021-05-14", test_setup_name="LEA_LK",
#                                             probe_dimensions="20x2x25 mm", measurement_method="Electrical", equipment_names="LeCroy HDO4104")

# N95 - LEA_LK
# create_permittivity_measurement_in_database("N95", "LEA_LK", company="Paderborn University", date="2021-05-11", test_setup_name="LEA_LK",
#                                             probe_dimensions="20x2x25 mm", measurement_method="Electrical", equipment_names="LeCroy HDO4104")

# custom_material
create_permittivity_measurement_in_database("custom_material", "custom_meas")

# clear_permittivity_measurement_data_in_database("N95", "LEA_LK")

frequency_list = [100000, 200000, 300000, 400000]
for T in [60, 100]:
    epsilon_r_tilde_list = []
    epsilon_phi_deg_list = []
    for f in frequency_list:
        epsilon_r_tilde, epsilon_phi_deg = get_permittivity_data_from_lea_lk(location, T, f, "N95")
        epsilon_r_tilde_list.append(np.mean(epsilon_r_tilde))
        epsilon_phi_deg_list.append(np.mean(epsilon_phi_deg))

    print(f"{frequency_list, epsilon_r_tilde_list, epsilon_phi_deg_list=}")
    # for each temperature store 3 lists
    # write_permittivity_data_into_database(T, frequency_list, epsilon_r_tilde_list, epsilon_phi_deg_list, "N95", "LEA_LK")
    write_permittivity_data_into_database(T, frequency_list, epsilon_r_tilde_list, epsilon_phi_deg_list, "custom_material", "custom_meas")
