"""Script to write permeability_data of datasheet data into the database only for TDK."""
from materialdatabase.material_data_base_functions import *
from materialdatabase.paths import *
import numpy as np


WRITE = False

material = str(Material.N27.value)
manufacturer = str(Manufacturer.TDK.value)
mu_r = 2000  # for calculation the value at b = 0

frequency = [10e3, 25e3, 50e3, 75e3, 100e3, 125e3, 150e3, 175e3, 200e3]
path_to_data = datasheet_path + "/" + manufacturer + "/" + material + "_digitized"

mu_amplitude_20C = read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, "permeability_over_b_field_20C_16kHz.csv"))
mu_amplitude_100C = read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, "permeability_over_b_field_100C_16kHz.csv"))

files_20C = ["powerloss_10kHz_20C.csv", "powerloss_25kHz_20C.csv", "powerloss_50kHz_20C.csv", "powerloss_75kHz_20C.csv", "powerloss_100kHz_20C.csv",
             "powerloss_125kHz_20C.csv", "powerloss_150kHz_20C.csv", "powerloss_175kHz_20C.csv", "powerloss_200kHz_20C.csv"]

# files_80C = ["powerloss_100kHz_80C.csv", "powerloss_200kHz_80C.csv", "powerloss_300kHz_80C.csv", "powerloss_400kHz_80C.csv", "powerloss_500kHz_80C.csv",
#              "powerloss_600kHz_80C.csv", "powerloss_700kHz_80C.csv", "powerloss_800kHz_80C.csv", "powerloss_900kHz_80C.csv", "powerloss_1000kHz_80C.csv"]

files_100C = ["powerloss_10kHz_100C.csv", "powerloss_25kHz_100C.csv", "powerloss_50kHz_100C.csv", "powerloss_75kHz_100C.csv", "powerloss_100kHz_100C.csv",
              "powerloss_125kHz_100C.csv", "powerloss_150kHz_100C.csv", "powerloss_175kHz_100C.csv", "powerloss_200kHz_100C.csv"]

b_25__200 = [0.025, 0.05, 0.1, 0.2]
# b_25__300 = [0.025, 0.05, 0.1, 0.2, 0.3]
# b_12_5__50 = [0.0125, 0.025, 0.05]
# b_12_5__100 = [0.0125, 0.025, 0.05, 0.1]

mu_r_25__200_20C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_20C[0], y_data=mu_amplitude_20C[1], x_new=b)) for b in np.array(b_25__200)*1000]
# mu_r_25__300_20C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_20C[0], y_data=mu_amplitude_20C[1], x_new=b)) for b in np.array(b_25__300)*1000]
# mu_r_12_5__100_25C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_25C[0], y_data=mu_amplitude_25C[1], x_new=b)) for b in np.array(b_12_5__100)*1000]

mu_r_25__200_100C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_100C[0], y_data=mu_amplitude_100C[1], x_new=b)) for b in np.array(b_25__200)*1000]
# mu_r_25__300_100C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_100C[0], y_data=mu_amplitude_100C[1], x_new=b)) for b in np.array(b_25__300)*1000]
# mu_r_12_5__50_100C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_100C[0], y_data=mu_amplitude_100C[1], x_new=b)) for b in np.array(b_12_5__50)*1000]
# mu_r_12_5__100_100C = [float(updates_x_ticks_for_graph(x_data=mu_amplitude_100C[0], y_data=mu_amplitude_100C[1], x_new=b))
#                        for b in np.array(b_12_5__100)*1000]

b_list = b_25__200
mu_values = mu_r_25__200_20C
permeability_data = []
for index, file in enumerate(files_20C):
    print(file)
    # if index == 3:  # no values for b=300mT and f>50kHz
    #     b_list = b_25__200
    #     mu_values = mu_r_25__200_20C

    data_dict = {"frequency": frequency[index],
                 "temperature": 25,
                 "flux_density": [0] + b_list,
                 "mu_r_abs": [mu_r] + mu_values,
                 "mu_phi_deg": [np.rad2deg(np.arctan(1/mu_r))] + [mu_phi_deg__from_mu_r_and_p_hyst(frequency=f, b_peak=b, mu_r=mu, p_hyst=p*1000)
                                                                  for b, f, mu, p in
                                                                  zip(b_list, read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[0],
                                                                      mu_values, read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[1])]
                 }
    permeability_data.append(data_dict)

# b_list = b_25__200
# mu_values = mu_r_25__200_100C
# for index, file in enumerate(files_80C):
#     if index == 3:  # no values for b=200mT and f>300kHz
#         b_list = b_25__100
#         mu_values = mu_r_25__100_100C
#     if index == 4:  # additonal values for b=12,5mT for f>500kHz
#         b_list = b_12_5__100
#         mu_values = mu_r_12_5__100_100C
#
#     data_dict = {"frequency": frequency[index],
#                  "temperature": 80,
#                  "flux_density": [0] + b_list,
#                  "mu_r_abs": [mu_r] + mu_values,
#                  "mu_phi_deg": [np.rad2deg(np.arctan(1/mu_r))] + [mu_phi_deg__from_mu_r_and_p_hyst(frequency=f, b_peak=b, mu_r=mu, p_hyst=p*1000)
#                                                                   for b, f, mu, p in
#                                                                   zip(b_list, read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[0],
#                                                                       mu_values,  read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[1])]
#                  }
#     permeability_data.append(data_dict)

b_list = b_25__200
mu_values = mu_r_25__200_100C
for index, file in enumerate(files_100C):
    # if index == 3:  # no values for b=300mT and f>50kHz
    #     b_list = b_25__200
    #     mu_values = mu_r_25__200_100C

    data_dict = {"frequency": frequency[index],
                 "temperature": 100,
                 "flux_density": [0] + b_list,
                 "mu_r_abs": [mu_r] + mu_values,
                 "mu_phi_deg": [np.rad2deg(np.arctan(1/mu_r))] + [mu_phi_deg__from_mu_r_and_p_hyst(frequency=f, b_peak=b, mu_r=mu, p_hyst=p*1000)
                                                                  for b, f, mu, p in
                                                                  zip(b_list, read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[0],
                                                                      mu_values, read_in_digitized_datasheet_plot(path=os.path.join(path_to_data, file))[1])]
                 }
    permeability_data.append(data_dict)

print(permeability_data)

if WRITE:

    with open(relative_path_to_db, "r") as jsonFile:
        database = json.load(jsonFile)

        database[material]["manufacturer_datasheet"]["permeability_data"] = permeability_data

    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(database, jsonFile, indent=2)
