"""Script to write data by Magnetics-TestBench into the material database."""
from materialdatabase.material_data_base_classes import *
from materialdatabase.paths import my_MTB_measurements_path
from datetime import date


# Control
write_data = True

# Set parameters
measurement_device = MeasurementDevice.zimmer
material_name = Material.DMR96A2
toroid_name = ToroidDirectoryName.DMR96A_2
create_permeability_measurement_in_database(Material.DMR96A2, MeasurementSetup.LEA_MTB, company=Company.UPB, date=str(date.today()),
                                            test_setup_name=MeasurementSetup.LEA_MTB, toroid_dimensions=toroid_name,
                                            measurement_method=MeasurementMethod.Electric, equipment_names=MeasurementDevice.ZESZimmer)

# General Path to measurements destination
mtb_post_pro_path = os.path.join(my_MTB_measurements_path, "post_processing_data_17_08_2023")
material_path = os.path.join(mtb_post_pro_path, measurement_device, "permeability", material_name)
frequencies = get_all_frequencies_for_material(material_path)

fig, ax = plt.subplots(3)
for frequency in frequencies:
    toroid_path = os.path.join(mtb_post_pro_path, measurement_device, "permeability", material_name, f"{int(frequency / 1000)}kHz", toroid_name)
    temperatures = get_all_temperatures_for_directory(toroid_path)
    for T in temperatures:
        file_path = os.path.join(toroid_path, str(T))

        b_ref, mu_r, mu_phi_deg = get_permeability_data_from_lea_mtb(file_path)

        b_reduced, f_p_hys_interpol_common, f_b_phi_interpol_common = interpolate_a_b_c(b_ref, mu_r, mu_phi_deg)
        b_ref, mu_r, mu_phi_deg = sort_data(b_reduced, f_p_hys_interpol_common, f_b_phi_interpol_common)

        b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref, mu_r, mu_phi_deg, smooth_data=True, crop_data=True, plot_data=True, ax=ax, f=frequency, T=T)

        if write_data:
            write_permeability_data_into_database(frequency, T, b_ref, mu_r, mu_phi_deg, material_name, MeasurementSetup.LEA_MTB)
plt.show()
