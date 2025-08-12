"""Collection of available materials and test setups.

The enums must be consistent with the FEM Magnetics Toolbox (FEMMT).
"""

from enum import Enum
from typing import Any
from materialdatabase.processing.utils.empirical import steinmetz_qT, enhanced_steinmetz_qT, \
    log_enhanced_steinmetz_qT, log_steinmetz_qT, \
    fit_mu_abs_TDK_MDT, fit_mu_abs_LEA_MTB, \
    fit_eps_qT


class FitFunction(str, Enum):
    """Set the usable fit function."""

    Steinmetz = "steinmetz"
    enhancedSteinmetz = "enhanced_steinmetz"
    mu_abs_TDK_MDT = "mu_abs_TDK_MDT"
    mu_abs_LEA_MTB = "mu_abs_LEA_MTB"
    eps_abs = "fit_eps_qT"

    def get_log_function(self) -> Any:
        """
        Get a the logarithmic callable function according to the defined enum.

        :return:
        """
        return {
            FitFunction.Steinmetz: log_steinmetz_qT,
            FitFunction.enhancedSteinmetz: log_enhanced_steinmetz_qT
        }[self]

    def get_function(self) -> Any:
        """
        Get a callable function according to the defined enum.

        :return:
        """
        return {
            FitFunction.Steinmetz: steinmetz_qT,
            FitFunction.enhancedSteinmetz: enhanced_steinmetz_qT,
            FitFunction.mu_abs_TDK_MDT: fit_mu_abs_TDK_MDT,
            FitFunction.mu_abs_LEA_MTB: fit_mu_abs_LEA_MTB,
            FitFunction.eps_abs: fit_eps_qT
        }[self]


class DatasheetCurvesFolder(str, Enum):
    """Set the name of the datasheet curves folder."""

    name = "datasheet_curves"


class DatasheetCurveType(str, Enum):
    """Set the type of datasheet curve."""

    mu_vs_b_at_T = "mu_vs_b_at_T"
    pv_vs_b_at_f_and_T = "pv_vs_b_at_f_and_T"
    pv_vs_f_at_b_and_T = "pv_vs_f_at_b_and_T"
    pv_vs_T_at_f_and_b = "pv_vs_T_at_f_and_b"


class ComplexDataType(str, Enum):
    """Set the type of complex material data."""

    complex_permeability = "complex_permeability"
    complex_permittivity = "complex_permittivity"


class MeasurementSetup(str, Enum):
    """Set the setup of the measurement."""

    LEA_MTB = "LEA_MTB"
    LEA_MTB_small_signal = "LEA_MTB_small_signal"
    LEA_LK = "LEA_LK"
    MagNet = "MagNet"
    TDK_MDT = "TDK_MDT"


class MeasurementMethod(str, Enum):
    """Set the method of the measurement."""

    ImpedanceAnalyzer = "Impedance Analyzer"
    Calorimetric = "Calorimetric"
    Electric = "Electric"
    Compensated = "Compensated"
    Direct = "Direct"


class Company(str, Enum):
    """Set the name of the company."""

    UPB = "Paderborn University"


class Manufacturer(str, Enum):
    """Set the name of the manufacturer."""

    TDK = "TDK"
    Ferroxcube = "Ferroxcube"
    DMEGC = "DMEGC"
    SUMIDA = "Sumida"
    FairRite = "Fair-Rite"
    Proterial = "Proterial"


class MeasurementDevice(str, Enum):
    """Set the type of the measurement device."""

    lecroy = "LeCroy_HDO4104"
    wayne_kerr = "Wayne_Kerr_6500B"
    zimmer = "ZES_Zimmer_LMG640"


class Material(str, Enum):
    """Set the name of the core material as enums."""

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
    Set the type of Permeability Measurement Probe.

    d_out: outer diameter of toroid in mm
    d_in: inner diameter of toroid in mm
    h: height of toroid in mm
    N1: primary turns number
    N2: secondary turns number
    """

    N87_1 = "R24,6x20,25x20,5_A00"
    DMR96A_1 = '???'
    DMR96A_2 = 'R_25.0x21.0x15.0x4x4'


class CuboidDirectoryName(str, Enum):
    """
    Set the type of Permittivity Measurement Probe.

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
