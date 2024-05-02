"""Write permeability data of datasheets into the database."""
from materialdatabase.material_data_base_classes import *
import materialdatabase.paths as paths
import datetime
import pandas as pd

# Control flags
write_data = True
plot_data = True

# Set parameters
core_name = 'R_14x9x5'  # d_out x d_in x h
material_name = Material._3F46
manufacturer = Manufacturer.Ferroxcube
date = str(datetime.datetime(2016, 3, 3))  # (year, month, day)
temperature_db = 100
frequencies_db = [1e6, 2e6, 3e6]

# Path to folder with datasheet data
datasheet_path = paths.datasheet_path

powerloss_data = []
powerloss_files = ["powerloss_density_over_b_field_1MHz_100C_new.csv",
                   "powerloss_density_over_b_field_2MHz_100C_new.csv",
                   "powerloss_density_over_b_field_3MHz_100C_new.csv"]

permeability_amplitude_data = []
permeability_amplitude_files = ["permeability_over_b_field_1MHz_100C_new.csv",
                                "placeholder.csv",  # placeholder because no data for 2MHz
                                "permeability_over_b_field_3MHz_100C_new.csv"]

b_field_data = []
permeability_angle_data = []

for index in range(len(powerloss_files)):
    print(index+1, "out of", len(powerloss_files), "calculated")

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    # adding the graph by 2MHz
    if index == 1:
        permeability_amplitude_dataframe_1 = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[0]), encoding="latin1")
        permeability_amplitude_dataframe_3 = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[2]), encoding="latin1")

        permeability_amplitude_dataframe_1_interpolated = updates_x_ticks_for_graph(x_data=permeability_amplitude_dataframe_1["x"],
                                                                                    y_data=permeability_amplitude_dataframe_1[" y"],
                                                                                    x_new=permeability_amplitude_dataframe_3["x"])

        b_field = permeability_amplitude_dataframe_3["x"]

        # calculating the mean of the graphs by 1MHz and 3 MHz for the missing graph of 2MHz
        permeability_mean = np.mean(np.array([permeability_amplitude_dataframe_1_interpolated, permeability_amplitude_dataframe_3[" y"]]), axis=0)

        permeability_amplitude_dataframe = pd.DataFrame(data={"x": b_field, " y": permeability_mean})

    else:
        permeability_amplitude_dataframe = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[index]), encoding="latin1")

    powerloss_dataframe = pd.read_csv(os.path.join(datasheet_path, powerloss_files[index]), encoding="latin1")

    permeability_interpolated = updates_x_ticks_for_graph(x_data=permeability_amplitude_dataframe["x"], y_data=permeability_amplitude_dataframe[" y"],
                                                          x_new=powerloss_dataframe["x"])

    b_reduced, f_p_hys_interpol_common, f_b_phi_interpol_common = interpolate_a_b_c(powerloss_dataframe["x"]/1000, permeability_interpolated,
                                                                                    mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequencies_db[index],
                                                                                                                     b_peak=powerloss_dataframe["x"]/1000,
                                                                                                                     mu_r=permeability_interpolated,
                                                                                                                     p_hyst=powerloss_dataframe[" y"]*1000))

    b_ref, mu_r, mu_phi_deg = sort_data(b_reduced, f_p_hys_interpol_common, f_b_phi_interpol_common)

    b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref_raw=b_ref, mu_r_raw=mu_r, mu_phi_deg_raw=mu_phi_deg, b_min=0.005, b_max=0.051,
                                                        smooth_data=False, crop_data=True, plot_data=True, ax=ax, f=frequencies_db[index], T=temperature_db)

    permeability_amplitude_data.append(mu_r)
    powerloss_data.append(powerloss_dataframe[" y"]*1000)
    b_field_data.append(b_ref)

    permeability_angle_data.append(mu_phi_deg)

    # Plot
    if plot_data:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[1].plot(b_ref, mu_phi_deg, label="angle")
        ax[0].plot(b_ref, mu_r, label="amplitude")

        ax[1].set_xlabel(PlotLabels.b_field.value)
        for ind in range(len(ax)):
            ax[ind].grid(True, which="both")
            ax[ind].legend()
        plt.show()


# Writing into material database
if write_data:

    flag_overwrite = True
    create_empty_material(material_name, manufacturer)
    create_permeability_measurement_in_database(material_name, measurement_setup="datasheet",
                                                company=manufacturer, date=date,
                                                test_setup_name="datasheet",
                                                toroid_dimensions=core_name,
                                                measurement_method="datasheet",
                                                equipment_names="datasheet", comment="")

    for index, value in enumerate(frequencies_db):

        write_permeability_data_into_database(frequency=value,
                                              temperature=temperature_db,
                                              b_ref=np.round(b_field_data[index], 3),
                                              mu_r_abs=np.round(permeability_amplitude_data[index], 3),
                                              mu_phi_deg=np.round(permeability_angle_data[index], 3),
                                              material_name=material_name,
                                              measurement_setup="datasheet",
                                              overwrite=flag_overwrite)
        flag_overwrite = False
