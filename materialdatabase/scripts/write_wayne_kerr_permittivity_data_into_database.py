"""Script to write permittivity data measured with the impedance analyzer into the database."""
# python libraries
from datetime import date

# own library imports
from materialdatabase.utils import get_closest, Z_from_amplitude_and_angle
from materialdatabase.material_data_base_classes import *
from materialdatabase.paths import your_sciebo

# 3rd party libraries
from scipy import constants

"""

    Setting of Impedance Analyzer
    Starting Frequency: 10kHz
    Ending Frequency: 8MHz
    Measuring Points: 800 points
    Temperatures: 30°C, 45°C, 60°C, 75°C

"""

# Control
write_data = True
plot_data = True

# Set parameters
core_name = "C_19.97x1.99x13.96"  # "C_19.98x1.98x13.97"  # b * a * h  CuboidDirectoryName._3F46_thin
core_dimensions = core_name[2:].split(sep="x")
material_name = Material.N27
manufacturer = Manufacturer.TDK
# measurements_path       = os.path.join(my_wayne_kerr_measurements_path, "small_signal", core_name, material_name)
measurements_path = os.path.join(your_sciebo, "Exchange_Sebastian", "08_Labor", "Messung_Permittivität", "N27")
temperatures_db = [30, 45, 60, 75]
frequencies_db = [10.1e3, 25e3, 50e3, 75e3, 1e5,  1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 4.5e5, 5e5, 5.5e5, 6e5, 6.5e5, 7e5, 7.5e5, 8e5, 8.5e5, 9e5, 9.5e5, 1e6,
                  1.1e6, 1.2e6, 1.3e6, 1.4e6, 1.5e6, 1.6e6, 1.7e6, 1.8e6, 1.9e6, 2e6, 2.1e6, 2.2e6, 2.3e6, 2.4e6, 2.5e6, 2.6e6, 2.7e6, 2.8e6, 2.9e6, 3e6,
                  3.1e6, 3.2e6, 3.3e6, 3.4e6, 3.5e6, 3.6e6, 3.7e6, 3.8e6, 3.9e6, 4e6, 4.1e6, 4.2e6, 4.3e6, 4.4e6, 4.5e6, 4.6e6, 4.7e6, 4.8e6, 4.9e6, 5e6,
                  5.1e6, 5.2e6, 5.3e6, 5.4e6, 5.5e6, 5.6e6, 5.7e6, 5.8e6, 5.9e6, 6e6]

if plot_data:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="all")

for temperature_db in temperatures_db:
    file_name = f"{temperature_db}.csv"

    # Read the impedance measurement data
    link_to_impedance_csv = [os.path.join(measurements_path, file_name)]
    print(f"{link_to_impedance_csv=}")
    f = np.genfromtxt(link_to_impedance_csv[0], delimiter=',', skip_header=True)[:, 0]
    Z_amplitude = np.genfromtxt(link_to_impedance_csv[0], delimiter=',', skip_header=True)[:, 1]
    phi_Z = np.genfromtxt(link_to_impedance_csv[0], delimiter=',', skip_header=True)[:, 2]
    Z = Z_from_amplitude_and_angle(Z_amplitude, phi_Z)

    # Dimensions of the probe cuboid capacitor
    h = float(core_dimensions[2]) / 1000
    a = float(core_dimensions[0]) / 1000
    b = float(core_dimensions[1]) / 1000
    A = a*b

    # Calculate permittivity data according to geometry and impedance
    eps_tilde_complex = h/(A * constants.epsilon_0 * 2 * np.pi * f * Z * complex(0, 1))
    eps_tilde_amplitude = abs(eps_tilde_complex)
    eps_tilde_angle = np.rad2deg(np.arccos(eps_tilde_complex.real/abs(eps_tilde_complex)))

    # Print the relevant values for the mdb
    indices = get_closest(frequencies_db, f)
    db_eps_tilde_amplitude = abs(eps_tilde_complex[indices])
    db_eps_tilde_angle = eps_tilde_angle[indices]

    # Plot
    if plot_data:
        ax[0].semilogx(f[indices], db_eps_tilde_amplitude, label=str(temperature_db) + core_name + " & " + material_name, marker="o", markersize=3.5)
        ax[1].semilogx(f[indices], db_eps_tilde_angle, label=str(temperature_db) + core_name + " & " + material_name, marker="o", markersize=3.5)
        ax[0].grid(True, which="both")
        ax[1].grid(True, which="both")
        ax[0].legend()
        ax[0].set_ylabel(r'$\tilde{\epsilon_\mathrm{r}}$')
        ax[1].set_ylabel(r'$\xi_\tilde{\epsilon_\mathrm{r}}$ in °')
        ax[1].set_xlim([min(f[indices])-1e3, max(f[indices])+0.1e6])
        plt.xlabel('frequency / Hz')
        plt.tight_layout()

    if write_data:
        # create_empty_material(material_name, manufacturer)
        create_permittivity_measurement_in_database(material_name, measurement_setup=MeasurementSetup.LEA_MTB_small_signal, company=Company.UPB,
                                                    date=str(date.today()), test_setup_name=MeasurementSetup.LEA_MTB_small_signal, probe_dimensions=core_name,
                                                    measurement_method=MeasurementMethod.ImpedanceAnalyzer, equipment_names=MeasurementDevice.wayne_kerr,
                                                    comment="")
        write_permittivity_data_into_database(temperature_db, list(f[indices]), list(db_eps_tilde_amplitude), list(db_eps_tilde_angle),
                                              material_name=material_name, measurement_setup=MeasurementSetup.LEA_MTB_small_signal)

if plot_data:
    plt.show()
