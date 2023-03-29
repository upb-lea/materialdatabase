from materialdatabase.material_data_base_functions import *
from materialdatabase.material_data_base_classes import *


location = "C:/Users/tpiepe/sciebo/Exchange_FEMMT/05_Materials/data/2022_10_10_Ferrite_mu_eps_Data_Keuck/"
# filename = create_file_name_LEA_LK()
# print(filename)

mdb = MaterialDatabase()

# Custom
# create_permeability_measurement_in_database("custom_material", "ABC")
# clear_permeability_measurement_data_in_database("custom_material", "ABC")

# N87 - LEA_LK (30, 60, 80, 100) x (100000, 200000, 300000, 400000)
# 500 kHz sieht nicht ordentlich aus
# create_permeability_measurement_in_database("N87", "LEA_LK", company="Paderborn University", date="2021-05-03", test_setup_name="LEA_LK",
#                                             toroid_dimensions="R24.6x20.1x24.5 mm", measurement_method="Electrical", equipment_names="Zimmer LMG640")

# N49 - LEA_LK (30, 60, 80, 100) x (100000, 200000, 300000, 400000, 500000)
# create_permeability_measurement_in_database("N49", "LEA_LK", company="Paderborn University", date="2021-04-29", test_setup_name="LEA_LK",
#                                             toroid_dimensions="R25x21x16 mm", measurement_method="Electrical", equipment_names="Zimmer LMG640")

create_permeability_measurement_in_database("N95", "LEA_LK", company="Paderborn University", date="2021-04-29", test_setup_name="LEA_LK",
                                            toroid_dimensions="R15x21x16 mm", measurement_method="Electrical", equipment_names="Zimmer LMG640")

for f in [100000, 200000, 300000, 400000, 500000]:
    for T in [30, 60, 80, 100]:
        print(f"{f, T =}")
        b_ref, mu_r, mu_phi_deg = get_permeability_data_from_lea_lk(location, f, T, "N95")

        b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref, mu_r, mu_phi_deg, smooth_data=True, crop_data=False, plot_data=True)

        # write_permeability_data_into_database(f, T, b_ref, mu_r, mu_phi_deg, "custom_material", "ABC")
        # write_permeability_data_into_database(f, T, b_ref, mu_r, mu_phi_deg, "N87", "LEA_LK")
        write_permeability_data_into_database(f, T, b_ref, mu_r, mu_phi_deg, "N95", "LEA_LK")
        # write_permeability_data_into_database(f, T, b_ref, mu_r, mu_phi_deg, "N49", "LEA_LK")
