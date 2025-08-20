"""Map e.g. functions to enums."""
from typing import Dict, Any
from materialdatabase.meta.data_enums import FitFunction, DataSource

MEASUREMENT_TO_FITFUNCTION: Dict[DataSource, FitFunction] = {
    DataSource.TDK_MDT: FitFunction.mu_abs_TDK_MDT,
    DataSource.LEA_MTB: FitFunction.mu_abs_LEA_MTB,
}

def get_fit_function_from_setup(setup: DataSource) -> Any:
    """
    Retrieve the corresponding fit function for a given measurement setup.

    This function maps a MeasurementSetup enum value to its associated
    FitFunction and returns the callable fit function used to model or fit
    permeability data (e.g., mu_abs_TDK_MDT, mu_abs_LEA_MTB, etc.).

    :param setup: MeasurementSetup enum value representing the measurement configuration.
    :return: Callable fit function associated with the setup.
    :raises ValueError: If no mapping is defined for the provided setup.
    """
    try:
        return MEASUREMENT_TO_FITFUNCTION[setup]
    except KeyError as err:
        raise ValueError(f"No fit function mapped for measurement setup: {setup}") from err
