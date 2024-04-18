"""Script to get permeability data by LEA_LK."""
from materialdatabase.material_data_base_functions import *

location = "C:/Users/tpiepe/sciebo/Exchange_FEMMT/05_Materials/data/2022_10_10_Ferrite_mu_eps_Data_Keuck/"
filename = create_permeability_file_name_lea_lk()

print(filename)
for f in [100000, 200000, 300000, 400000, 500000]:
    for T in [30, 60, 80, 100]:

        b_ref, mu_r, mu_phi_deg = get_permeability_data_from_lea_lk(location, f, T, "N49")

        b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref, mu_r, mu_phi_deg, smooth_data=True, crop_data=False, plot_data=True)
