from enum import IntEnum, Enum


# Following Enums must always be consistent with the FEM Magnetics Toolbox (FEMMT)
class MaterialDataSource(str, Enum):
    """Sets the source from where data is taken.
    """
    Custom = "custom"
    Measurement = "measurements"
    ManufacturerDatasheet = "manufacturer_datasheet"


class MeasurementDataType(str, Enum):
    """Sets the type of measurement data.
    """
    ComplexPermeability = "complex_permeability"
    ComplexPermittivity = "complex_permittivity"
    Steinmetz = "Steinmetz"
