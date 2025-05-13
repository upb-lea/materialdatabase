"""Collection of available materials and test setups.

The enums must be consistent with the FEM Magnetics Toolbox (FEMMT).
"""

from enum import Enum


class MeasurementSetup(str, Enum):
    """Sets the setup of the measurement."""

    LEA_MTB = "LEA_MTB"
    LEA_MTB_small_signal = "LEA_MTB_small_signal"
    LEA_LK = "LEA_LK"
    MagNet = "MagNet"


class MeasurementMethod(str, Enum):
    """Sets the method of the measurement."""

    ImpedanceAnalyzer = "Impedance Analyzer"
    Calorimetric = "Calorimetric"
    Electric = "Electric"
    Compensated = "Compensated"
    Direct = "Direct"


class Company(str, Enum):
    """Sets the name of the company."""

    UPB = "Paderborn University"


class Manufacturer(str, Enum):
    """Sets the name of the manufacturer."""

    TDK = "TDK"
    Ferroxcube = "Ferroxcube"
    DMEGC = "DMEGC"
    SUMIDA = "Sumida"
    FairRite = "Fair-Rite"
    Proterial = "Proterial"


class MeasurementDevice(str, Enum):
    """Sets the type of the measurement device."""

    lecroy = "LeCroy_HDO4104"
    wayne_kerr = "Wayne_Kerr_6500B"
    zimmer = "ZES_Zimmer_LMG640"


class CuboidCodeNames(str, Enum):
    """Probe-Codes of permittivity measurement probes."""

    # 3F46
    D966 = "C25,16x2,04x15,55"
    T227 = "C25,05x4,05x15,78"

    # DMR96A
    M667 = "C24,91x2,19x16,25"
    A266 = "C24,98x1,94x15,01"

    # N95
    R211 = "C31,89x5,01x22,25"
    P926 = "C31,18x5,04x22,05"
    W276 = "C30,6x4,99x21,95"
    D978 = "C31,95x5,02x22,16"
    D133 = "C40,33x2,03x21,04"
    L551 = "C40,02x3,02x20,04"

    # N87
    X979 = "C25,04x2x21,57"
    H382 = "C25,07x2,02x21,55"

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        """
        Check if key is part of class.

        :param key: variable to check
        :type key: str
        """
        return key in cls.__members__


class ToroidCodeNames(str, Enum):
    """Probe-Codes of permeability measurement probes."""

    # 3F46
    A461 = "R36x32x16"
    C293 = "R24x18x15"

    # DMR96A
    N133 = "R24,96x20,91x15,03"
    E725 = "R24,93x20,89x12,01"

    # N95
    E376 = "R25x21x16,1"
    P141 = "R20.97x18.99x16.12"
    T149 = "R23x20x16,1"

    # N87
    U175 = "R24,7x20,3x20,5"

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        """
        Check if key is part of class.

        :param key: variable to check
        :type key: str
        """
        return key in cls.__members__


class Material(str, Enum):
    """Sets the name of the core material as enums."""

    TEST = "TEST"  # FOR TESTING STUFF
    _3F46 = "3F46"
    _3C90 = "3C90"
    _3C92 = "3C92"
    _3C94 = "3C94"
    _3C95 = "3C95"
    _3E6 = "3E6"
    _3F4 = "3F4"
    _77 = "77"
    _78 = "78"
    _79 = "79"
    ML95S = "ML95S"
    T37 = "T37"
    N27 = "N27"
    N30 = "N30"
    N49 = "N49"
    N87 = "N87"
    N95 = "N95"
    PC200 = "PC200"
    custom_material = "custom_material"
    DMR96A = "DMR96A"
    DMR96A2 = "DMR96A2"
    DMR96A2_test = "DMR96A2_test"


class ToroidDirectoryName(str, Enum):
    """
    Sets the type of Permeability Measurement Probe.

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
    """
    Sets the type of Permittivity Measurement Probe.

    string with:
    a x b x c

    a: width of cuboid in mm
    b: thickness of cuboid in mm
    c: height of cuboid in mm

    (a and b can be exchanged, because A = a*b)
    """

    _3F46_thin = "C_25.16x2.04x15.55"
    DMR96A_2 = "C_25x2x15"
    # DMR96A_2 = 'R_25.0x21.0x15.0x4x4'


class MagNetFileNames(str, Enum):
    """Name of the MagNet .mat-files with the raw data."""

    _3C90 = "3C90_TX-25-15-10_Data1_Cycle.mat"
    _3C92 = "3C92_TX25-25-12_Data2_Cycle.mat"
    _3C94 = "3C94_TX-20-10-7_Data1_Cycle.mat"
    _3C95 = "3C95_TX25-25-12_Data2_Cycle.mat"
    _3E6 = "3E6_TX-22-14-6.4_Data1_Cycle.mat"
    _3F4 = "3F4_E-32-6-20-R_Data1_Cycle.mat"
    _77 = "77_0014_Data1_Cycle.mat"
    _78 = "78_0076_Data1_Cycle.mat"
    _79 = "79_1801_Data12_Cycle.mat"
    _ML95S = "ML95S_OR-14-5-8H_Data2_Cycle.mat"
    _N27 = "N27_R20.0X10.0X7.0_Data1_Cycle.mat"
    _N30 = "N30_22.1X13.7X6.35_Data1_Cycle.mat"
    _N49 = "N49_R16.0X9.6X6.3_Data1_Cycle.mat"
    _N87 = "N87_R34.0X20.5X12.5_Data5_Cycle.mat"
    _T37 = "T37_TX25X14.8X10_Data1_Cycle.mat"
