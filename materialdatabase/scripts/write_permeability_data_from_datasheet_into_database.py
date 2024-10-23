import numpy as np
from materialdatabase.material_data_base_classes import *
from materialdatabase.material_data_base_functions import *
from materialdatabase.paths import *

# Control flags
WRITE = True

material = str(Material._79.value)
manufacturer = str(Manufacturer.FairRite.value)

temperature = [25, 25, 25, 25, 25, 100, 100, 100, 100, 100]
frequency = [100e3, 300e3, 500e3, 750e3, 1000e3, 100e3, 300e3, 500e3, 750e3, 1000e3]

amplitude_permeability_flux_density_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                                        "amplitude_permeability_flux_density_25C_10kHz.csv"))
amplitude_permeability_flux_density_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                                         "amplitude_permeability_flux_density_100C_10kHz.csv"))

# 25C
powerloss_100kHz_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                     "powerloss_flux_density_25C_100kHz.csv"))
powerloss_300kHz_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                     "powerloss_flux_density_25C_300kHz.csv"))
powerloss_500kHz_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                     "powerloss_flux_density_25C_500kHz.csv"))
powerloss_750kHz_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                     "powerloss_flux_density_25C_750kHz.csv"))
powerloss_1000kHz_25C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                      "powerloss_flux_density_25C_1000kHz.csv"))
# 100C
powerloss_100kHz_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                      "powerloss_flux_density_100C_100kHz.csv"))
powerloss_300kHz_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                      "powerloss_flux_density_100C_300kHz.csv"))
powerloss_500kHz_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                      "powerloss_flux_density_100C_500kHz.csv"))
powerloss_750kHz_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                      "powerloss_flux_density_100C_750kHz.csv"))
powerloss_1000kHz_100C = read_in_digitized_datasheet_plot(os.path.join(datasheet_path, manufacturer, material + "_digitized",
                                                                       "powerloss_flux_density_100C_1000kHz.csv"))
# 25C
amplitude_permeability_flux_density_100kHz_25C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_25C[0],
                                                                           y_data=amplitude_permeability_flux_density_25C[1],
                                                                           x_new=powerloss_100kHz_25C[0], kind="linear")
amplitude_permeability_flux_density_300kHz_25C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_25C[0],
                                                                           y_data=amplitude_permeability_flux_density_25C[1],
                                                                           x_new=powerloss_300kHz_25C[0], kind="linear")
amplitude_permeability_flux_density_500kHz_25C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_25C[0],
                                                                           y_data=amplitude_permeability_flux_density_25C[1],
                                                                           x_new=powerloss_500kHz_25C[0], kind="linear")
amplitude_permeability_flux_density_750kHz_25C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_25C[0],
                                                                           y_data=amplitude_permeability_flux_density_25C[1],
                                                                           x_new=powerloss_750kHz_25C[0], kind="linear")
amplitude_permeability_flux_density_1000kHz_25C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_25C[0],
                                                                            y_data=amplitude_permeability_flux_density_25C[1],
                                                                            x_new=powerloss_1000kHz_25C[0], kind="linear")
# 100C
amplitude_permeability_flux_density_100kHz_100C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_100C[0],
                                                                            y_data=amplitude_permeability_flux_density_100C[1],
                                                                            x_new=powerloss_100kHz_100C[0], kind="linear")
amplitude_permeability_flux_density_300kHz_100C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_100C[0],
                                                                            y_data=amplitude_permeability_flux_density_100C[1],
                                                                            x_new=powerloss_300kHz_100C[0], kind="linear")
amplitude_permeability_flux_density_500kHz_100C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_100C[0],
                                                                            y_data=amplitude_permeability_flux_density_100C[1],
                                                                            x_new=powerloss_500kHz_100C[0], kind="linear")
amplitude_permeability_flux_density_750kHz_100C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_100C[0],
                                                                            y_data=amplitude_permeability_flux_density_100C[1],
                                                                            x_new=powerloss_750kHz_100C[0], kind="linear")
amplitude_permeability_flux_density_1000kHz_100C = updates_x_ticks_for_graph(x_data=amplitude_permeability_flux_density_100C[0],
                                                                             y_data=amplitude_permeability_flux_density_100C[1],
                                                                             x_new=powerloss_1000kHz_100C[0], kind="linear")
# 25C
mu_phi_deg_100kHz_25C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[0], b_peak=np.array(powerloss_100kHz_25C[0])/10000,
                                                         mu_r=amplitude_permeability_flux_density_100kHz_25C, p_hyst=np.array(powerloss_100kHz_25C[1])*1000)
mu_phi_deg_300kHz_25C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[1], b_peak=np.array(powerloss_300kHz_25C[0])/10000,
                                                         mu_r=amplitude_permeability_flux_density_300kHz_25C, p_hyst=np.array(powerloss_300kHz_25C[1])*1000)
mu_phi_deg_500kHz_25C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[2], b_peak=np.array(powerloss_500kHz_25C[0])/10000,
                                                         mu_r=amplitude_permeability_flux_density_500kHz_25C, p_hyst=np.array(powerloss_500kHz_25C[1])*1000)
mu_phi_deg_750kHz_25C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[3], b_peak=np.array(powerloss_750kHz_25C[0])/10000,
                                                         mu_r=amplitude_permeability_flux_density_750kHz_25C, p_hyst=np.array(powerloss_750kHz_25C[1])*1000)
mu_phi_deg_1000kHz_25C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[4], b_peak=np.array(powerloss_1000kHz_25C[0])/10000,
                                                          mu_r=amplitude_permeability_flux_density_1000kHz_25C, p_hyst=np.array(powerloss_1000kHz_25C[1])*1000)
# 100C
mu_phi_deg_100kHz_100C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[0], b_peak=np.array(powerloss_100kHz_100C[0])/10000,
                                                          mu_r=amplitude_permeability_flux_density_100kHz_100C, p_hyst=np.array(powerloss_100kHz_100C[1])*1000)
mu_phi_deg_300kHz_100C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[1], b_peak=np.array(powerloss_300kHz_100C[0])/10000,
                                                          mu_r=amplitude_permeability_flux_density_300kHz_100C, p_hyst=np.array(powerloss_300kHz_100C[1])*1000)
mu_phi_deg_500kHz_100C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[2], b_peak=np.array(powerloss_500kHz_100C[0])/10000,
                                                          mu_r=amplitude_permeability_flux_density_500kHz_100C, p_hyst=np.array(powerloss_500kHz_100C[1])*1000)
mu_phi_deg_750kHz_100C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[3], b_peak=np.array(powerloss_750kHz_100C[0])/10000,
                                                          mu_r=amplitude_permeability_flux_density_750kHz_100C, p_hyst=np.array(powerloss_750kHz_100C[1])*1000)
mu_phi_deg_1000kHz_100C = mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequency[4], b_peak=np.array(powerloss_1000kHz_100C[0])/10000,
                                                           mu_r=amplitude_permeability_flux_density_1000kHz_100C, p_hyst=np.array(powerloss_1000kHz_100C[1])*1000)

# print(mu_phi_deg_25kHz)
# print(mu_phi_deg_50kHz)
# print(mu_phi_deg_100kHz)
# print(mu_phi_deg_200kHz)

mu_r_list = [amplitude_permeability_flux_density_100kHz_25C, amplitude_permeability_flux_density_300kHz_25C, amplitude_permeability_flux_density_500kHz_25C,
             amplitude_permeability_flux_density_750kHz_25C, amplitude_permeability_flux_density_1000kHz_25C,
             amplitude_permeability_flux_density_100kHz_100C, amplitude_permeability_flux_density_300kHz_100C, amplitude_permeability_flux_density_500kHz_100C,
             amplitude_permeability_flux_density_750kHz_100C, amplitude_permeability_flux_density_1000kHz_100C]
mu_phi_deg_list = [mu_phi_deg_100kHz_25C, mu_phi_deg_300kHz_25C, mu_phi_deg_500kHz_25C, mu_phi_deg_750kHz_25C, mu_phi_deg_1000kHz_25C,
                   mu_phi_deg_100kHz_100C, mu_phi_deg_300kHz_100C, mu_phi_deg_500kHz_100C, mu_phi_deg_750kHz_100C, mu_phi_deg_1000kHz_100C]
flux_density = [np.array(powerloss_100kHz_25C[0])/10000, np.array(powerloss_300kHz_25C[0])/10000, np.array(powerloss_500kHz_25C[0])/10000,
                np.array(powerloss_750kHz_25C[0])/10000, np.array(powerloss_1000kHz_25C[0])/10000,
                np.array(powerloss_100kHz_100C[0])/10000, np.array(powerloss_300kHz_100C[0])/10000, np.array(powerloss_500kHz_100C[0])/10000,
                np.array(powerloss_750kHz_100C[0])/10000, np.array(powerloss_1000kHz_100C[0])/10000]


if WRITE:
    with open(relative_path_to_db, "r") as jsonFile:
        database = json.load(jsonFile)
else:
    database = {material: {"manufacturer_datasheet": {}}}

data_list = []
for index, value in enumerate(frequency):
    print(frequency[index])
    print(temperature[index])
    # print(np.array(flux_density[index]))
    # print(mu_r_list[index])
    # print(mu_phi_deg_list[index])
    b_ref, mu_r, mu_phi_deg = process_permeability_data(b_ref_raw=np.array(flux_density[index]), mu_r_raw=mu_r_list[index],
                                                        mu_phi_deg_raw=mu_phi_deg_list[index], b_min=0.005, b_max=0.31,
                                                        smooth_data=False, crop_data=False, plot_data=True, f=frequency[index], T=temperature[index])
    print(mu_phi_deg)
    data_dict = {"mu_r_abs": list(mu_r),
                 "mu_phi_deg": list(mu_phi_deg),
                 "flux_density": list(b_ref),
                 "frequency": frequency[index],
                 "temperature": temperature[index]}
    data_list.append(data_dict)

database[material]["manufacturer_datasheet"]["permeability_data"] = data_list

if WRITE:
    with open(relative_path_to_db, "w") as jsonFile:
        json.dump(database, jsonFile, indent=2)
