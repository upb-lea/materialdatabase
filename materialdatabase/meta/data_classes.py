"""Collection of dataclasses."""
from dataclasses import dataclass
import numpy as np

from materialdatabase.meta.data_enums import Material, DataSource, FitFunction


@dataclass
class ComplexPermeabilityConfig:
    """Configuration container for a magnetic material."""

    material: Material
    setup: DataSource
    pv_fit_function: FitFunction


@dataclass
class ComplexPermeabilityPlotConfig:
    """Configuration container for a magnetic material to be included in the plots."""

    mat_cfg: ComplexPermeabilityConfig
    enabled: bool
    label: str
    color: str
    marker: str

