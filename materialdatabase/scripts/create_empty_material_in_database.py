"""Script to create empty material in database."""
from materialdatabase.material_data_base_classes import *


# Control
WRITE = True

# Set parameters
material_name = Material._3F4
manufacturer = Manufacturer.Ferroxcube
initial_permeability = 2300
resistivity = 5
max_flux_density = 0.47
volumetric_mass_density = 4800


if WRITE:
    create_empty_material(material_name=material_name, manufacturer=manufacturer, initial_permeability=initial_permeability, resistivity=resistivity,
                          max_flux_density=max_flux_density, volumetric_mass_density=volumetric_mass_density)
