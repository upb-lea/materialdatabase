"""Collection of dataclasses."""
from dataclasses import dataclass

from data_enums import Material, MeasurementSetup


@dataclass
class MaterialPlotConfig:
    """Configuration container for a magnetic material to be included in the plots."""

    enabled: bool
    material: Material
    setup: MeasurementSetup
    label: str
    color: str
    marker: str
