"""Definition of the data structure and data/measurement/... types.

The enums must be consistent with the FEM Magnetics Toolbox (FEMMT).
"""

from enum import Enum


class MaterialDataSource(str, Enum):
    """Sets the source from where data is taken."""

    Custom = "custom"
    Measurement = "measurements"
    ManufacturerDatasheet = "manufacturer_datasheet"


class MeasurementDataType(str, Enum):
    """Sets the type of the measurement data."""

    ComplexPermeability = "complex_permeability"
    ComplexPermittivity = "complex_permittivity"
    Steinmetz = "Steinmetz"


class DatasheetPlotName(str, Enum):
    """Sets the type of the datasheet plot."""

    complex_permeability = "complex_permeability"
    initial_permeability_temperature = "initial_permeability_temperature"
    initial_permeability_frequency = "initial_permeability_frequency"
    incremental_permeability_field_strength = "incremental_permeability_field_strength"
    amplitude_permeability_flux_density = "amplitude_permeability_flux_density"
    b_h_curve = "b_h_curve"
    relative_core_loss_frequency = "relative_core_loss_frequency"
    relative_core_loss_flux_density = "relative_core_loss_flux_density"
    relative_core_loss_temperature = "relative_core_loss_temperature"
    permeability_data = "permeability_data"


class MeasurementParameter(str, Enum):
    """Sets the type of the measurement parameter."""

    signal_shape = "signal_shape"
    temperature = "temperature"
    frequency = "frequency"
    H_DC_offset = "H_DC_offset"
    flux_density = "flux_density"
    field_strength = "field_strength"
    mu_r_abs = "mu_r_abs"
    mu_phi_deg = "mu_phi_deg"
    epsilon_r = "epsilon_r"
    epsilon_phi_deg = "epsilon_phi_deg"


class PlotLabels(str, Enum):
    """Labels for axes of possible plots."""

    time = "time in s"
    time_ms = "time in ms"
    time_us = "time in µs"

    frequency_Hz = "frequency in Hz"
    frequency_kHz = "frequency in kHz"
    frequency_MHz = "frequency in MHz"

    temperature_in_C = "temperature in °C"
    temperature_in_K = "temperature in K"

    powerloss_density_mW = r"powerloss density  in mW/cm^3"
    powerloss_density_kW = r"powerloss density  in kW/m^3"
    powerloss_density_W = r"powerloss density  in W/m^3"

    h_field = "magnetic field strength in A/m"
    b_field = "magnetic flux density in T"
    b_field_mT = "magnetic flux density in mT"
    e_field = "electric field strength in V/m"
    d_field = r"electric flux density in As/m^2"

    current_A = "voltage in A"
    current_mA = "voltage in mA"
    voltage_V = "current in V"
    voltage_mV = "current in mV"
    power_W = "power in W"
    power_mW = "power in mW"

    mu_ampl = r"rel. permeability amplitude $\mu_\mathrm{r}}$"
    mu_angle = r"rel. permeability angle $\mu_\mathrm{r}}$ in degree"
    mu_init = r"rel. permeability initial $\mu_\mathrm{r}}$"
    eps_ampl = r"rel. permittivity amplitude  $\tilde{\epsilon}_\mathrm{r}}$"
    eps_angle = r"rel. permittivity angle  $\tilde{\epsilon}_\mathrm{r}}$ in degree"


class HeaderMeasurementData(str, Enum):
    """
    Names for the header of the dataframes for the post-processing files(Magnetic-TestBench).

    e.g. data of the permeability angle gets the header "permeability angle"
    """

    frequency = "frequency"
    time = "time"

    powerloss_density = "powerloss density"

    mag_flux_density = "magnetic flux density"
    mag_field_strength = "magnetic field strength"

    elec_flux_density = "electric flux density"
    elec_field_strength = "electric field strength"

    permeability_ampl = "permeability amplitude"
    permeability_angle = "permeability angle"

    permittivity_ampl = "permittivity amplitude"
    permittivity_angle = "permittivity angle"

    self_inductance = "self inductance"
    prim_leak_inductance = "primary leakage inductance"
    sec_leak_inductance = "secondary leakage inductance"

    voltage = "voltage"
    current = "current"
    power = "power"

    impedance_ampl = "impedance amplitude"
    impedance_angle = "impedance angle"

    impedance_ampl_prim_open = "primary open impedance amplitude"
    impedance_angle_prim_open = "primary open impedance angle"

    impedance_ampl_prim_short = "primary short impedance amplitude"
    impedance_angle_prim_short = "primary short impedance angle"

    impedance_ampl_sec_open = "secondary open impedance amplitude"
    impedance_angle_sec_open = "secondary open impedance angle"


class ProbeDimensions(str, Enum):
    """Sets the name for the dimensions of a probe."""

    height = "height"
    cross_section = "cross section"
    volume = "volume"

    width = "width"
    thickness = "thickness"

    out_diameter = "outer diameter"
    inn_diameter = "inner diameter"

    prim_winding = "primary winding"
    sec_winding = "secondary winding"

    l_mag = "lmag"
