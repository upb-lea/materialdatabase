"""Script to write permeability data by MagNet into the material database."""
from materialdatabase.material_data_base_classes import *
import pandas as pd
import os
from materialdatabase.paths import my_MagNet_data_path


# load MagNet data
mtb_path = os.path.join(my_MagNet_data_path, "N87.csv")
df = pd.read_csv(mtb_path)
f_sine_lim_clustered_set = sorted(set(df["frequency"]))
min_number_of_measurements = 10

# Init the database entry
create_permeability_measurement_in_database("N87", "MagNet", company="Princeton", date="tba", test_setup_name="MagNet",
                                            toroid_dimensions="tba", measurement_method="tba", equipment_names="tba")


fig, ax = plt.subplots(3)
for frequency in f_sine_lim_clustered_set:
    if len(df[df["frequency"] == frequency]["b"]) > min_number_of_measurements:
        b_ref, mu_r, mu_phi_deg = np.array(df[df["frequency"] == frequency]["b"]), np.array(df[df["frequency"] == frequency]["mu_r_abs"]), \
            np.array(df[df["frequency"] == frequency]["mu_phi_deg"])

        b_ref, mu_r, mu_phi_deg = sort_data(b_ref, mu_r, mu_phi_deg)
        b_ref, mu_r, mu_phi_deg = interpolate_a_b_c(b_ref, mu_r, mu_phi_deg)
        b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref, mu_r, mu_phi_deg, smooth_data=True, crop_data=True, plot_data=True, ax=ax, f=frequency)
        # write_permeability_data_into_database(frequency, 25, b_ref, mu_r, mu_phi_deg, "N87", "MagNet")

plt.show()
