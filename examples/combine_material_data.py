"""Example to combine material data.

This script enables users to combine material data, if this is still no done.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

import materialdatabase as mdb
from materialdatabase import get_user_colors as colors
from materialdatabase.processing.plot import plot_combined_loss, plot_mu_all, StyleDict
from materialdatabase.meta.data_classes import ComplexPermeabilityPlotConfig, ComplexPermeabilityConfig



# ---------------------------------------------
# Configuration
# ---------------------------------------------

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Flags to control which plots to generate
PLOT_MU_ABS = True
PLOT_PV = True

# Operating points of interest
FREQS = np.linspace(2e5, 4e5, 8)  # Frequency range in Hertz
FLUX_DENSITIES = np.linspace(0.14, 0.07, 8)  # Flux densities in Tesla
TEMPS = np.ones_like(FREQS) * 100  # Temperatures in Celsius

# Materials to evaluate
mat_cfg=ComplexPermeabilityConfig(
            material=mdb.Material.N27,
            # setup=mdb.DataSource.TDK_MDT,
            setup=mdb.DataSource.MagNet,
            pv_fit_function=mdb.FitFunction.enhancedSteinmetz
        )

# ---------------------------------------------
# Load Material Data
# ---------------------------------------------

mdb_data = mdb.Data()

mdb_data.combine_material_permeability_data(material=mat_cfg.material, data_source=mat_cfg.setup)


# combine_material_permeability_data(act_root_dir: str, material: Material, data_source: DataSource) -> bool:


