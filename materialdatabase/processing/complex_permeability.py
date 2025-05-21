"""Class to represent the data structure and load material data."""
import logging
# python libraries

# 3rd party libraries
import pandas as pd

# own libraries
from materialdatabase.meta.data_enums import *
from materialdatabase.meta.config import *

logger = logging.getLogger(__name__)


class ComplexPermeability:
    """Class to process complex permeability data."""

    def __init__(self, df_complex_permeability: pd.DataFrame, material: Material, measurement_setup: MeasurementSetup):
        """
        Initialize the complex permeability measurement data.

        :param df_complex_permeability: pd.DataFrame with header ["f", "T", "b", "mu_real", "mu_imag"]
        :param material: e.g. mdb.Material.N95
        :param measurement_setup: e.g. mdb.MeasurementSetup.TDK_MDT
        """
        self.measurement_data = df_complex_permeability
        self.material = material
        self.measurement_setup = measurement_setup
