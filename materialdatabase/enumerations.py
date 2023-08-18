from enum import Enum


# Following Enums must always be consistent with the FEM Magnetics Toolbox (FEMMT)
class MaterialDataSource(str, Enum):
    """Sets the source from where data is taken.
    """
    Custom = "custom"
    Measurement = "measurements"
    ManufacturerDatasheet = "manufacturer_datasheet"


class MeasurementSetup(str, Enum):
    """Sets the source from where data is taken.
    """
    LEA_MTB = "LEA_MTB"
    LEA_MTB_small_signal = "LEA_MTB_small_signal"
    LEA_LK = "LEA_LK"


class MeasurementMethod(str, Enum):
    """Sets the source from where data is taken.
    """
    ImpedanceAnalyzer = "Impedance Analyzer"
    Calorimetric = "Calorimetric"
    PrimaryCurrentSecondaryVoltage = "Primary Current and Secondary Voltage"


class Manufacturer(str, Enum):
    """Sets the source from where data is taken.
    """
    TDK = "TDK"
    Ferroxcube = "Ferroxcube"
    DMEGC = "DMEGC"


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
    DMR96A = "DMR96A"
    DMR96A2 = "DMR96A2"
    DMR96A2_test = "DMR96A2_test"


class ToroidDirectoryName(str, Enum):
    """Sets the type of Permeability Measurement Device

    string with:
    d_out x d_in x h x N1 x N2

    d_out: outer diameter of toroid in mm
    d_in: innter diameter of toroid in mm
    h: height of toroid in mm
    N1: primary turns number
    N2: secondary turns number
    """
    N87_1 = "R24,6x20,25x20,5_A00"
    DMR96A_1 = '???'
    DMR96A_2 = 'R_25.0x21.0x15.0x4x4'


class CuboidDirectoryName(str, Enum):
    """Sets the type of Permittivity Measurement Device

    string with:
    a x b x c

    a: width of cuboid in mm
    b: thickness of cuboid in mm
    c: height of cuboid in mm

    (a and b can be exchanged, because A = a*b)
    """
    DMR96A_2 = "C_25.0x2.0x15.0"

