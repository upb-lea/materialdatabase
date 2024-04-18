"""Dataclass for storing material curves."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MaterialCurve:
    """
    Stores material curves for one material together in this dataclass.

    This dataclass can be loaded into a calculation, by only choosing the material.
    All other parameters are bound to this material.
    """

    material_name: str
    material_mu_r_abs: float
    material_flux_density_vec: np.ndarray
    material_mu_r_imag_vec: np.ndarray
    material_mu_r_real_vec: np.ndarray
    saturation_flux_density: float
    boundary_temperature: float
    boundary_frequency: float
