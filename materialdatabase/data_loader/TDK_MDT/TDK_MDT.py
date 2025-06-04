"""Script to load material data from the TDK Magnetic Design Tool in the material database."""

import logging
from pathlib import Path

import materialdatabase as mdb
from functions_TDK_MDT import tdkmdt2pandas

# Configure logging to show femmt terminal output
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Define path to external TDK_MDT data
link_to_data = Path(mdb.get_user_paths().external_material_data).joinpath("TDK_MDT")

# Frequency data in kHz
frequencies = [25, 50, 100, 200, 300, 500, 700, 1000]

# Loss density limit in W/m³
pv_max = 3e6

# Process N49
temperatures_n49 = [25, 30, 40, 50, 60, 70, 80, 90, 100]
tdkmdt2pandas(
    link_to_data,
    mdb.Material.N49,
    frequencies,
    temperatures_n49,
    save2file=True,
    pv_max=pv_max,
)

# Process N87
temperatures_n87 = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
tdkmdt2pandas(
    link_to_data,
    mdb.Material.N87,
    frequencies,
    temperatures_n87,
    save2file=True,
    pv_max=pv_max,
)

# Process N95
temperatures_n95 = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
tdkmdt2pandas(
    link_to_data,
    mdb.Material.N95,
    frequencies,
    temperatures_n95,
    save2file=True,
    pv_max=pv_max,
)
