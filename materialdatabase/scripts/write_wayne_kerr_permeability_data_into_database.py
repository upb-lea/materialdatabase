"""Script to write permeability data by the impedance analyzer into the material database."""
import matplotlib.pyplot as plt
from scipy import constants
from materialdatabase.utils import L_from_Z, get_closest
from materialdatabase.paths import my_wayne_kerr_measurements_path
from materialdatabase.material_data_base_classes import *
from datetime import date

# Control
write_data = False
plot_data = True

# Set parameters
core_name = ToroidDirectoryName.DMR96A_2  # d_out x d_in x h x N1 x N2
core_dimensions = core_name[2:].split(sep="x")
material_name = Material.DMR96A2
manufacturer = Manufacturer.DMEGC
file_names = ["lk.csv", "l11.csv", "l22.csv"]
temperature_db = 25
frequencies_db = [50e3, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6, 1.1e6, 1.2e6]

measurements_path = os.path.join(my_wayne_kerr_measurements_path, "small_signal", core_name, material_name)


links = [os.path.join(measurements_path, file_names[0]),
         os.path.join(measurements_path, file_names[1]),
         os.path.join(measurements_path, file_names[2])]

Z_k = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 1]
phi_k = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 2]
Z_11 = np.genfromtxt(links[1], delimiter=',', skip_header=True)[:, 1]
phi_11 = np.genfromtxt(links[1], delimiter=',', skip_header=True)[:, 2]
Z_22 = np.genfromtxt(links[2], delimiter=',', skip_header=True)[:, 1]
phi_22 = np.genfromtxt(links[2], delimiter=',', skip_header=True)[:, 2]
f = np.genfromtxt(links[0], delimiter=',', skip_header=True)[:, 0]

# Real valued inductances
Lk = L_from_Z(Z_k, phi_k, f)
L11 = L_from_Z(Z_11, phi_11, f)
L22 = L_from_Z(Z_22, phi_22, f)

# T-ECD primary concentrated
Ls = Lk
k = np.sqrt(1 - Lk.real / L11.real)
M = k * np.sqrt(L11 * L22)
# n = M.real / L22.real
Lm = k ** 2 * L11
# print(f"{k = }, {Ls = }, {M = }, {Lm = }")


# Dimensions of the probe (toroid)
d_outer = float(core_dimensions[0]) / 1000
d_inner = float(core_dimensions[1]) / 1000
h = float(core_dimensions[2]) / 1000
N1 = int(core_dimensions[3])
N2 = int(core_dimensions[4])
w = 0.5 * (d_outer - d_inner)

# TECD
Z_11_complex = Z_11 * (np.cos(np.deg2rad(phi_11)) + complex(0, 1) * np.sin(np.deg2rad(phi_11)))
Z_22_complex = Z_22 * (np.cos(np.deg2rad(phi_22)) + complex(0, 1) * np.sin(np.deg2rad(phi_22)))
Z_k_complex = Z_k * (np.cos(np.deg2rad(phi_k)) + complex(0, 1) * np.sin(np.deg2rad(phi_k)))
n = N1/N2
Z_s1 = 0.5 * (Z_11_complex - Z_22_complex*n**2 + Z_k_complex)
Z_m_complex_alt = Z_11_complex - Z_s1
# print(Z_m_complex_alt)

# Self impedance
Z_11_complex = Z_11 * (np.array(np.cos(np.deg2rad(phi_11)) + j * np.sin(np.deg2rad(phi_11))))
mu_r_complex = 1 / N1**2 * (2 * np.pi * f * h / (2 * np.pi) * np.log(d_outer / d_inner))**(-1) * \
    np.array(Z_11_complex.imag + j * Z_11_complex.real) / constants.mu_0

# Magnetization impedance
Z_m_complex = j*2*np.pi*f*Lm
mu_r_complex_m = 1 / N1**2 * (2 * np.pi * f * h / (2 * np.pi) * np.log(d_outer / d_inner))**(-1) * \
    np.array(Z_m_complex.imag + j * Z_m_complex.real) / constants.mu_0

# Alternative magnetization impedance
# Z_m_complex_alt = 1 / N1**2 * Z_11_complex - Z_k * N1 * (np.array(np.cos(np.deg2rad(phi_k)) + j * np.sin(np.deg2rad(phi_k))))
mu_r_complex_m_alt = 1 / N1**2 * (2 * np.pi * f * h / (2 * np.pi) * np.log(d_outer / d_inner))**(-1) * \
    np.array(Z_m_complex_alt.imag + j * Z_m_complex_alt.real) / constants.mu_0

# plt.plot(f, abs(Z_11_complex)-abs(Z_11_complex), label="z11")
# plt.plot(f, (abs(Z_11_complex)-abs(Z_m_complex))/2/np.pi/f, label="zs1")
# plt.plot(f, (abs(Z_11_complex)-abs(Z_m_complex_alt))/2/np.pi/f, label="zs1 alt")
# plt.legend()
# plt.show()


# Print the relevant values for the mdb
indices = get_closest(frequencies_db, f)
print(f"Chosen Frequencies for database: {frequencies_db}")
print(f"Frequencies for database: {f[indices]}")

# Permeability from taking self inductance
# print(abs(mu_r_complex[indices]))
# print(np.rad2deg(np.arctan(mu_r_complex[indices].imag/mu_r_complex[indices].real)))
# Permeability from magnetizing inductance
# print(abs(mu_r_complex_m[indices]))
# print(np.rad2deg(np.arctan(mu_r_complex_m[indices].imag/mu_r_complex_m[indices].real)))
# Permeability from alternative magnetizing approach
# print(abs(mu_r_complex_m_alt[indices]))
# print(np.rad2deg(np.arctan(mu_r_complex_m_alt[indices].imag / mu_r_complex_m_alt[indices].real)))

# Plot
if plot_data:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(f[indices], abs(mu_r_complex[indices]), label="abs")
    ax[0].plot(f[indices], abs(mu_r_complex_m[indices]), label="abs_m")
    ax[0].plot(f[indices], abs(mu_r_complex_m_alt[indices]), label="abs_m_alternative")
    ax[1].plot(f[indices], np.rad2deg(np.arctan(mu_r_complex[indices].imag / mu_r_complex[indices].real)), label="arc")
    ax[1].plot(f[indices], np.rad2deg(np.arctan(mu_r_complex_m[indices].imag / mu_r_complex_m[indices].real)), label="arc_m")
    ax[1].plot(f[indices], np.rad2deg(np.arctan(mu_r_complex_m_alt[indices].imag / mu_r_complex_m_alt[indices].real)), label="arc_m_alt")
    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    ax[0].set(xlabel='Frequency in Hz', ylabel=r'rel. Permeability')
    plt.show()


# Writing into material database
if write_data:

    # create virtual b_ref for small signal
    b_min = 0           # always!
    b_max = 0.001       # assume very small value to allow interpolation but without taking into account amplitude dependency
    b_ref = [b_min, b_max]

    db_mu_r_abs = abs(mu_r_complex_m[indices])
    db_mu_phi_deg = np.rad2deg(np.arctan(mu_r_complex_m[indices].imag / mu_r_complex_m[indices].real))

    flag_overwrite = True
    create_empty_material(material_name, manufacturer)
    create_permeability_measurement_in_database(material_name, measurement_setup=MeasurementSetup.LEA_MTB_small_signal, company=Company.UPB,
                                                date=str(date.today()), test_setup_name=MeasurementSetup.LEA_MTB_small_signal, toroid_dimensions=core_name,
                                                measurement_method=MeasurementMethod.ImpedanceAnalyzer, equipment_names=MeasurementDevice.WayneKerr, comment="")

    for i, frequency in enumerate(f[indices]):

        write_permeability_data_into_database(frequency=frequency, temperature=temperature_db, b_ref=b_ref, mu_r_abs=[db_mu_r_abs[i], db_mu_r_abs[i]],
                                              mu_phi_deg=[db_mu_phi_deg[i], db_mu_phi_deg[i]], material_name=material_name,
                                              measurement_setup=MeasurementSetup.LEA_MTB_small_signal, overwrite=flag_overwrite)
        flag_overwrite = False
