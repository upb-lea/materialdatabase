"""Load permittivity data in the mdb."""
import logging

from pathlib import Path
from os.path import isfile, isdir
from os import listdir

import pandas as pd

import materialdatabase as mdb
from materialdatabase.meta.data_enums import Material

logger = logging.getLogger(__name__)

# Name of the ferrite material
material_name = str(Material.N49.value)
# Probe code of the toroid probe
probe_code = "65Y"

# Get paths from config.toml and join them
path2processed_data = Path(mdb.get_user_paths().external_material_data).joinpath("LEA_MTB").joinpath("permeability").joinpath("processed")\
    .joinpath(material_name).joinpath(probe_code)
path2mdb_data = Path(mdb.get_user_paths().material_data).joinpath("complex_permeability").joinpath("LEA_MTB")

# Check if path exists.
if isdir(path2processed_data):
    # Get all files in given directory.
    files_in_path2processed_data = [f for f in listdir(path2processed_data) if isfile(path2processed_data.joinpath(str(f)))]

    # Check if only one file is in specified path
    if len(files_in_path2processed_data) == 1:
        # read in measurement data
        df = pd.read_csv(path2processed_data.joinpath(str(files_in_path2processed_data[0])), encoding="latin1")
        # select right columns and store processed data in csv file
        df[["f", "T", "b", "mu_real", "mu_imag"]].to_csv(path2mdb_data.joinpath(f"{material_name}.csv"), index=False)
        logger.info("Measurement of Probe" + probe_code + " made of the material " + material_name + " written in material database.")
    else:
        if len(files_in_path2processed_data) == 0:
            logger.warning("No file in path", path2processed_data)
        else:
            logger.warning("Too many files in path", path2processed_data)
else:
    logger.warning("Path " + str(path2processed_data) + " does not exist!")
