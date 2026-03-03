"""Load permeability data in the mdb."""
import logging

from pathlib import Path
from os.path import isfile, isdir
from os import listdir

import pandas as pd

import materialdatabase as mdb
from materialdatabase.meta.data_enums import Material

logger = logging.getLogger(__name__)

# configure logging to show terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# - - - - - - - - - - - - - - - -
# Define paths and load probe data
# - - - - - - - - - - - - - - - -

def create_paths_to_LEA_MTB_measurements_from_config():
    """
    Create paths to permeability measurements from config file.

    :return: path2processed_measurement_data and path2probe_database
    """
    # Path to the directory, where the measurement data from the LEA Magnetics Testbench are located
    path2measurement_data = Path(mdb.get_user_paths().external_material_data).joinpath("LEA_MTB")

    # Path to the directory of the processed measurement data
    path2processed_measurement_data = path2measurement_data.joinpath("permeability_temporary").joinpath("processed")

    # Path to the file, where all probes' data is stored centrally
    path2probe_database = path2measurement_data.joinpath("probe_database.csv")

    return path2processed_measurement_data, path2probe_database


# Paths to the measurement data of the LEA Magnetics testbench
path2processed_measurement_data, path2probe_database = create_paths_to_LEA_MTB_measurements_from_config()

# Read in the probe database as a pd.DataFrame
df_probe_database = pd.read_csv(path2probe_database)
print(df_probe_database)

# Path to the directory, where the data shall be stored inside the materialdatabase
path2mdb_data = Path(mdb.get_user_paths().material_data).joinpath("complex_permeability").joinpath("LEA_MTB")


# - - - - - - - - - - - - - - - -
# Iterate over materials:
# - - - - - - - - - - - - - - - -

# Name of the ferrite material to be processed
material_name = Material._3F46.value

# Path to the directory, where the processed measurement data of the specified material is located
path2processed_material_data = path2processed_measurement_data.joinpath(material_name)

# Check if path exists.
if not isdir(path2processed_material_data):
    logger.warning("Path " + str(path2processed_material_data) + " does not exist!")

else:
    # Find the directories inside the folder where the material data is stored
    directories_in_path2processed_data = [f for f in listdir(path2processed_material_data) if isdir(path2processed_material_data.joinpath(str(f)))]
    logger.info(f"Found {len(directories_in_path2processed_data)} directories (probe names) in {path2processed_material_data} : "
                f"{directories_in_path2processed_data}")

    # initialize an empty dataframe
    df_material = pd.DataFrame()

    # Iterate over probes
    for directory in directories_in_path2processed_data:

        # Check whether the directory name matches any entry in the probe_database.csv
        if directory in df_probe_database["probe_code"].values:
            probe_code = directory
        else:
            logger.warning(f"The directory name {directory} does not match any probe name in the probe_database.csv \n"
                           f"-> ({path2probe_database})")
            break

        # create path to the directory of the probe's measurement data
        path2processed_probe_data = path2processed_material_data.joinpath(probe_code)

        # Get all files in given directory.
        files_in_path2processed_probe_data = [f for f in listdir(path2processed_probe_data) if isfile(path2processed_probe_data.joinpath(str(f)))]

        # If only a single file is in the specified path, assume that it contains the correct measurement:
        if len(files_in_path2processed_probe_data) != 1:
            if len(files_in_path2processed_probe_data) == 0:
                logger.warning("No file in path", path2processed_probe_data)
            else:
                logger.warning("Too many files in path", path2processed_probe_data)
        else:
            # read in the measurement data
            df_probe = pd.read_csv(path2processed_probe_data.joinpath(str(files_in_path2processed_probe_data[0])), encoding="latin1")

            # add probe column (since it's not in the CSV)
            df_probe["probe"] = probe_code

            # enforce final header order
            df_probe = df_probe[["probe", "f", "T", "b", "mu_real", "mu_imag"]]

            # append to material dataframe
            df_material = pd.concat([df_material, df_probe], ignore_index=True)

            logger.info(f"Write permeability data of material {material_name} measured on probe {probe_code} into the materialdatabase.")

        # Check if reading the data was successful
        if df_material.empty:
            logger.warning('DataFrame is empty!')
        else:
            # Store processed data in csv file
            df_material.to_csv(path2mdb_data.joinpath(f"{material_name}.csv"), index=False)
