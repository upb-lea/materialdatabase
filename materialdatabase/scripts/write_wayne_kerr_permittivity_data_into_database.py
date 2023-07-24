from scipy import constants
from materialdatabase.utils import L_from_Z, get_closest, Z_from_amplitude_and_angle
from materialdatabase.material_data_base_classes import *
from materialdatabase.paths import my_wayne_kerr_measurements_path

write_data = False
plot_data = True

j = complex(0, 1)

core_name = "C_25x2x21.6"  # b * a * h
core_dimensions = core_name[2:].split(sep="x")
print(core_dimensions)
material_name = "N87"
signal_type = "small_signal"
measurements_path = os.path.join(my_wayne_kerr_measurements_path, "small_signal", core_name, material_name)

# Definition
temperature = 25
frequencies = [50e3, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6, 1.1e6, 1.2e6]


# links = [os.path.join(measurements_path, "Z_phi.csv")]
links = [os.path.join(measurements_path, "quaderfullrange.csv")]
# links = [os.path.join(measurements_path, "Z_phi_evening.csv")]
# links = [os.path.join(measurements_path, "quadealterkontaktr.csv")]


f = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 0]
Z_amplitude = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 1]
phi_Z = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 2]

Z = Z_from_amplitude_and_angle(Z_amplitude, phi_Z)

# Dimensions of the probe (toroid)
h = float(core_dimensions[2]) / 1000
a = float(core_dimensions[0]) / 1000
b = float(core_dimensions[1]) / 1000
A = a*b

# permittivity data
eps_tilde_complex = h/(A * constants.epsilon_0 * 2 * np.pi * f * Z * complex(0, 1))
eps_tilde_amplitude = abs(eps_tilde_complex)
eps_tilde_angle = np.rad2deg(np.arccos(eps_tilde_complex.real/abs(eps_tilde_complex)))

# Plot
if plot_data:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="all")
    ax[0].semilogx(f, eps_tilde_amplitude, label=core_name + " & " + material_name)
    ax[1].semilogx(f, eps_tilde_angle, label=core_name + " & " + material_name)
    ax[0].grid(True, which="both")
    ax[1].grid(True, which="both")
    ax[0].legend()
    ax[0].set_ylabel(r'rel. Ersatz-Permittivität $\tilde{\epsilon_\mathrm{r}}$')
    ax[1].set_ylabel(r'Winkel $\delta_\tilde{\epsilon_\mathrm{r}}$ / °')
    plt.xlabel('Frequenz / Hz')
    plt.tight_layout()
    plt.show()


if write_data:
    # Print the relevant values for the mdb
    indices = get_closest(frequencies, f)
    print(abs(eps_tilde_complex[indices]))
    print(eps_tilde_angle[indices])

    db_eps_tilde_amplitude = abs(eps_tilde_complex[indices])
    db_eps_tilde_angle = eps_tilde_angle[indices]

    # Write to mdb
    mdb = MaterialDatabase()
    create_permittivity_measurement_in_database(material_name, signal_type)
    write_permittivity_data_into_database(temperature, list(f[indices]), list(db_eps_tilde_amplitude), list(db_eps_tilde_angle), material_name, signal_type)
