from enum import Enum


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


class MeasurementDevice(str, Enum):
    """Sets the type of Measurement Device
    """
    ZESZimmer = "ZES-Zimmer_LMG640"
    WayneKerr = "Wayne_Kerr_6500B"
    LeCroy = "LeCroy_HDO4104"


class Material(str, Enum):
    """Sets the type of Measurement Device
    """
    N87 = "N87"


class ToroidDirectoryName(str, Enum):
    """Sets the type of Measurement Device
    """
    N87_1 = "R24,6x20,25x20,5_A00"

