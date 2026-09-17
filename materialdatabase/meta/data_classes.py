"""Collection of dataclasses."""

from dataclasses import dataclass

from materialdatabase.meta.data_enums import Material, DataSource, FitFunction
from materialdatabase.processing.complex_permeability import LossFitModel, PermeabilityFitModel


@dataclass
class ComplexPermeabilityConfig:
    """Configuration container for a magnetic material."""

    material: Material
    setup: DataSource
    pv_fit_function: FitFunction | LossFitModel
    probe_codes: None | list[str] = None
    mu_a_fit_function: FitFunction | PermeabilityFitModel | None = None


@dataclass
class ComplexPermeabilityPlotConfig:
    """Configuration container for a magnetic material to be included in the plots."""

    mat_cfg: ComplexPermeabilityConfig
    enabled: bool
    label: str
    color: str
    marker: str
    line_style: str = "-"
