"""Script to load material data from the TDK Magnetic Design Tool in the material database."""

import logging
from pathlib import Path
import materialdatabase as mdb
from functions_TDK_MDT import tdkmdt2pandas

# Configure logging to show detailed output
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Define path to external TDK_MDT data
link_to_data = Path(mdb.get_user_paths().external_material_data).joinpath("TDK_MDT")

# Global limit for loss density in W/m³
pv_max = 3e6

# Electable materials' configuration
# Each entry defines the material, whether it should be processed, and its temperature/frequency configuration
materials_to_process = {
    "PC200": {
        "enabled": True,
        "material": mdb.Material.PC200,
        "frequencies": [700, 1000],  # kHz
        "temperatures": [25, 30, 40, 50, 60, 70, 80, 90, 100],  # Celsius
    },
    "N49": {
        "enabled": True,
        "material": mdb.Material.N49,
        "frequencies": [25, 50, 100, 200, 300, 500, 700, 1000],
        "temperatures": [25, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    "N87": {
        "enabled": True,
        "material": mdb.Material.N87,
        "frequencies": [25, 50, 100, 200, 300, 500, 700, 1000],
        "temperatures": [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    },
    "N95": {
        "enabled": True,
        "material": mdb.Material.N95,
        "frequencies": [25, 50, 100, 200, 300, 500, 700, 1000],
        "temperatures": [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    },
    # Add more materials ...
}


def process_materials():
    """Process all enabled materials using the tdkmdt2pandas conversion."""
    for name, config in materials_to_process.items():
        if config["enabled"]:
            logging.info(f"Processing material: {name}")
            tdkmdt2pandas(
                link_to_data,
                config["material"],
                config["frequencies"],
                config["temperatures"],
                save2file=True,
                pv_max=pv_max,
            )
        else:
            logging.info(f"Skipping material: {name}")


if __name__ == "__main__":
    process_materials()
