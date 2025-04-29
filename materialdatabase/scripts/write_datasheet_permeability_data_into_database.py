"""
Script to write permeability_data of datasheet data into the database, only for the ferrite material 3F46 of Ferroxcube.

The data under the key "permeability_data" is used for the FEM-simulations in FEMMT. Based on the data in the datasheet, the amplitude and angle of the
permeability is calculated for different magnetic flux density values.
"""

from materialdatabase.material_data_base_classes import *
from materialdatabase.material_data_base_functions import *
import materialdatabase.paths as paths
import datetime

# Control flags
WRITE = False
PLOT = True

# Set parameters
core_name = 'R_14x9x5'  # d_out x d_in x h
material_name = Material._3F46
manufacturer = Manufacturer.Ferroxcube
date = str(datetime.datetime(2016, 3, 3))  # (year, month, day)
temperature_db = 100
frequencies_db = [1e6, 2e6, 3e6]

# Path to folder with datasheet data
datasheet_path = paths.datasheet_path + "Ferroxcube/3F46_digitized/"
powerloss_data = []
powerloss_files = ["powerloss_flux_density_100C_1000kHz.csv",
                   "powerloss_flux_density_100C_2000kHz.csv",
                   "powerloss_flux_density_100C_3000kHz.csv"]

permeability_amplitude_data = []
permeability_amplitude_files = ["amplitude_permeability_100C_1000kHz.csv",
                                "placeholder.csv",  # placeholder because no data for 2MHz
                                "amplitude_permeability_100C_3000kHz.csv"]

b_field_data = []
permeability_angle_data = []

for index in range(len(powerloss_files)):
    print(index+1, "out of", len(powerloss_files), "calculated")

    # adding the graph by 2MHz
    if index == 1:
        permeability_amplitude_df_1 = read_in_digitized_datasheet_plot(path=os.path.join(datasheet_path, permeability_amplitude_files[0]))
        permeability_amplitude_df_3 = read_in_digitized_datasheet_plot(path=os.path.join(datasheet_path, permeability_amplitude_files[2]))

        permeability_amplitude_df_3_interpolated = updates_x_ticks_for_graph(x_data=permeability_amplitude_df_3[0], y_data=permeability_amplitude_df_3[1],
                                                                             x_new=permeability_amplitude_df_1[0])
        b_field = permeability_amplitude_df_1[0]

        # calculating the mean of the graphs by 1MHz and 3 MHz for the missing graph of 2MHz
        permeability_mean = np.mean(np.array([permeability_amplitude_df_1[1], permeability_amplitude_df_3_interpolated]), axis=0)

        permeability_amplitude_df = [b_field, permeability_mean]

    else:
        permeability_amplitude_df = read_in_digitized_datasheet_plot(path=os.path.join(datasheet_path, permeability_amplitude_files[index]))

    powerloss_df = read_in_digitized_datasheet_plot(path=os.path.join(datasheet_path, powerloss_files[index]))

    min_b = max([min(powerloss_df[0]), min(permeability_amplitude_df[0])])
    max_b = min([max(powerloss_df[0]), max(permeability_amplitude_df[0])])
    b_common = np.linspace(min_b, max_b, 20)

    permeability_updated = updates_x_ticks_for_graph(x_data=permeability_amplitude_df[0], y_data=permeability_amplitude_df[1], x_new=b_common)
    powerloss_updated = updates_x_ticks_for_graph(x_data=powerloss_df[0], y_data=powerloss_df[1], x_new=b_common)

    mu_phi_deg = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequencies_db[index], b_peak=b_common/1000, mu_r=permeability_updated,
                                                  p_hyst=np.array(powerloss_updated)*1000)

    b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref_raw=b_common/1000, mu_r_raw=permeability_updated, mu_phi_deg_raw=mu_phi_deg,
                                                        b_min=0.005, b_max=0.051, smooth_data=False, crop_data=False, plot_data=PLOT,
                                                        f=frequencies_db[index], T=temperature_db)

    permeability_amplitude_data.append(mu_r)
    powerloss_data.append(np.array(powerloss_updated)*1000)
    b_field_data.append(b_ref)
    permeability_angle_data.append(mu_phi_deg)

    # Plot
    if PLOT:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax[1].plot(b_ref, mu_phi_deg, label="angle")
        ax[0].plot(b_ref, mu_r, label="amplitude")
        ax[1].set_xlabel(PlotLabels.b_field.value)

        for ind in range(len(ax)):
            ax[ind].grid(True, which="both")
            ax[ind].legend()
        plt.show()


if WRITE:
    with open(relative_path_to_db, "r") as jsonFile:
        database = json.load(jsonFile)
else:
    database = {material_name.value: {"manufacturer_datasheet": {}}}

data_list = []
for index, value in enumerate(frequencies_db):
    print(value)
    print(temperature_db)

    data_dict = {"mu_r_abs": list(np.round(permeability_amplitude_data[index], 3)),
                 "mu_phi_deg": list(np.round(permeability_angle_data[index], 3)),
                 "flux_density": list(np.round(b_field_data[index], 3)),
                 "frequency": frequencies_db[index],
                 "temperature": temperature_db}
    data_list.append(data_dict)

database[material_name.value]["manufacturer_datasheet"]["permeability_data"] = data_list

if WRITE:
    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(database, jsonFile, indent=2)
