from scipy import constants
from materialdatabase.utils import get_closest, Z_from_amplitude_and_angle
from materialdatabase.material_data_base_classes import *
# from materialdatabase.paths import my_wayne_kerr_measurements_path
import datetime
import pandas as pd

# Control
write_data = False
plot_data = True
create_material = False

# Set parameters
core_name = 'R_14x9x5'  # d_out x d_in x h x N1 x N2
core_dimensions = core_name[2:].split(sep="x")
material_name = Material._3F46
manufacturer = Manufacturer.Ferroxcube
date = datetime.datetime(2016, 3, 3)  # (year, month, day)
temperature_db = 100
frequencies_db = [1e6, 2e6, 3e6]

datasheet_path = "C:/Users/schacht/sciebo/Exchange_Sebastian/05_Material_Datasheets/Ferroxcube/3F46_digitized/"

powerloss_data = []
powerloss_files = ["powerloss_density_over_b_field_1MHz_100C.csv",
                   "powerloss_density_over_b_field_2MHz_100C.csv",
                   "powerloss_density_over_b_field_3MHz_100C.csv"]


permeability_amplitude_data = []
permeability_amplitude_files = ["permeability_over_b_field_1MHz_100C.csv",
                                "placeholder.csv",
                                "permeability_over_b_field_3MHz_100C.csv"]

b_field_data = []
permeability_angle_data = []

for index, value in enumerate(powerloss_files):
    print(index)

    # adding the graph by 2MHz
    if index == 1:
        permeability_amplitude_dataframe_1 = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[0]), encoding="latin1")
        permeability_amplitude_dataframe_3 = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[2]), encoding="latin1")

        f_linear = interp1d(permeability_amplitude_dataframe_1["x"], permeability_amplitude_dataframe_1[" y"], fill_value="extrapolate")

        permeability_amplitude_dataframe_1_interpolated = f_linear(permeability_amplitude_dataframe_3["x"])

        b_field = permeability_amplitude_dataframe_3["x"]
        permeability_interpolated = np.mean(np.array([permeability_amplitude_dataframe_1_interpolated, permeability_amplitude_dataframe_3[" y"]]), axis=0)

        permeability_amplitude_dataframe = pd.DataFrame(data={"x": b_field, " y": permeability_interpolated})

    else:
        permeability_amplitude_dataframe = pd.read_csv(os.path.join(datasheet_path, permeability_amplitude_files[index]), encoding="latin1")

    powerloss_dataframe = pd.read_csv(os.path.join(datasheet_path, powerloss_files[index]), encoding="latin1")

    f_linear = interp1d(permeability_amplitude_dataframe["x"], permeability_amplitude_dataframe[" y"])

    permeability_interpolated = f_linear(powerloss_dataframe["x"])

    permeability_amplitude_data.append(permeability_interpolated)
    powerloss_data.append(powerloss_dataframe[" y"])
    b_field_data.append(powerloss_dataframe["x"])

    permeability_angle_data.append(mu_phi_deg__from_mu_r_and_p_hyst(frequency=frequencies_db[index], b_peak=b_field_data[index]/1000,
                                                                    mu_r=permeability_amplitude_data[index], p_hyst=powerloss_data[index]*1000))


# Plot
if plot_data:
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax[0].plot(b_field_data[1], permeability_angle_data[1], label="angle")
    ax[1].plot(b_field_data[1], permeability_amplitude_data[1], label="amplitude")
    ax[2].plot(b_field_data[1], powerloss_data[1], label="powerloss")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
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
                                              b_ref=b_field_data[index],
                                              mu_r_abs=permeability_amplitude_data[index],
                                              mu_phi_deg=permeability_angle_data[index],
                                              material_name=material_name,
                                              measurement_setup="datasheet",
                                              overwrite=flag_overwrite)
        flag_overwrite = False
