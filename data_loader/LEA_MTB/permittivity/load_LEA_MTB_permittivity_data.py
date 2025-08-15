"""Load permittivity data in the mdb."""
from pathlib import Path

import numpy as np
import pandas as pd

import materialdatabase as mdb
from functions_LEA_MTB_permittivity import collect_probes_and_temperatures
# from functions_permittivity import collect_probes_and_temperatures, enrich_with_probe_dimensions, process_permittivity


# get paths from config.toml
path2processed_data = Path(mdb.get_user_paths().external_material_data).joinpath("LEA_MTB").joinpath("permittivity").joinpath("processed")
path2mdb_data = Path(mdb.get_user_paths().material_data).joinpath()

measurement_structure = collect_probes_and_temperatures(path2processed_data)


# Iterative reading of the measurements
for material_name, probes in measurement_structure.items():
    for probe_code, data in probes.items():
        path2processed_probe = path2processed_data.joinpath(material_name).joinpath(probe_code)
        df_material = pd.DataFrame()  # start empty

        for temperature in data["temperatures"]:
            print(f"Material={material_name}, Probe={probe_code}, Temp={temperature}")

            # Path to the impedance measurement at a specific temperature
            path2processed_file = path2processed_probe.joinpath(f"{temperature}.csv")

            df_temp = pd.read_csv(path2processed_file)

            # add temperature column (since it's not in the CSV)
            df_temp["T"] = temperature

            # convert angle to radians
            angle_rad = np.deg2rad(df_temp["permittivity angle"])

            # calculate eps_real and eps_imag
            df_temp["eps_real"] = df_temp["permittivity amplitude"] * np.cos(angle_rad)
            df_temp["eps_imag"] = df_temp["permittivity amplitude"] * np.sin(angle_rad)

            # rename frequency to f
            df_temp = df_temp.rename(columns={"frequency": "f"})

            # enforce final header order
            df_temp = df_temp[["f", "T", "eps_real", "eps_imag"]]

            # append to material dataframe
            df_material = pd.concat([df_material, df_temp], ignore_index=True)

        # Assume there is only one probe per material
        path2mdb_material_file = (
            path2mdb_data.joinpath("complex_permittivity")
            .joinpath("LEA_MTB")
            .joinpath(f"{material_name}.csv")
        )

        # store processed data in csv file
        df_material.to_csv(path2mdb_material_file, index=False)
