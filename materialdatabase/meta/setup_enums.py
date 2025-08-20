"""Definition of the data structure and data/measurement/... types.

The enums must be consistent with the FEM Magnetics Toolbox (FEMMT).
"""

from enum import Enum


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
