"""Script to get the property from LEA_LK."""
from materialdatabase.material_data_base_functions import *
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as savgol

location = "C:/Users/tpiepe/sciebo/Exchange_FEMMT/05_Materials/data/2022_10_10_Ferrite_mu_eps_Data_Keuck/"

filename = create_permeability_file_name_lea_lk()

print(filename)
for f in [200000, 300000]:
    b_hys, p_hys = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="Core_Loss", quantity="p_hys", frequency=f,
                                                         material_name="N49", temperature=30)
    b_phi, mu_phi_deg = get_permeability_property_from_lea_lk(path_to_parent_folder=location, sub_folder_name="mu_phi_Plot", quantity="mu_phi", frequency=f,
                                                              material_name="N49", temperature=30)
    if len(b_phi) != len(b_hys):
        print("invalid")
        break
    b_ref = b_phi

    # pre, end = 10, 5
    # b_ref = crop_data_fixed(b_ref, pre, end)
    # p_hys = crop_data_fixed(p_hys, pre, end)
    # mu_phi_deg = crop_data_fixed(mu_phi_deg, pre, end)

    # smooth input data
    mu_r = mu_r__from_p_hyst_and_mu_phi_deg(mu_phi_deg, f, b_ref, p_hys)
    mu_r_smoothed = savgol(x=mu_r, window_length=10, polyorder=2)
    # mu_r_smoothed = savgol(x=mu_r_smoothed, window_length=10, polyorder=2)
    plt.plot(b_ref, mu_r)
    plt.plot(b_ref, mu_r_smoothed)
    # plt.plot(b_phi, mu_phi_deg)

plt.grid()
plt.show()
