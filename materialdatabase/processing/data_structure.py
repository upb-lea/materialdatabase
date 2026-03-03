"""Class to represent the data structure and load material data."""
import logging
# python libraries
from pathlib import Path, PurePath
from typing import Any

# 3rd party libraries
import toml
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# own libraries
from materialdatabase.meta.data_enums import ComplexDataType, Material, DataSource, FitFunction, \
    DatasheetCurveType, DatasheetCurvesFolder, DatasheetAttribute
from materialdatabase.meta.config import check_paths_in_toml, get_user_paths
from materialdatabase.processing.complex_permeability import ComplexPermeability
from materialdatabase.processing.complex_permittivity import ComplexPermittivity

logger = logging.getLogger(__name__)


class Data:
    """Represent the structure of a folder tree containing CSV files."""

    def __init__(self, root_dir: str = get_user_paths().material_data):
        """
        Initialize the Structure by scanning a directory.

        :param root_dir: The base directory to scan for CSV files.
        """
        self.root_dir = Path(root_dir).resolve()
        self.config_dir = Path(self.root_dir.resolve().parent / "meta" / "config.toml")
        self.structure = self._scan_structure()
        self.all_paths = self.get_all_paths()

    def _scan_structure(self) -> dict:
        """
        Scan the folder tree for CSV files.

        :return: A dictionary mapping relative folder paths to lists of CSV files.
        """
        check_paths_in_toml()

        structure: dict[str, dict[str, list[str]]] = {}
        root_path = Path(self.root_dir)

        for path in root_path.rglob("*.csv"):
            if path.is_file():
                rel_dir = path.parent.relative_to(root_path)
                structure.setdefault(str(rel_dir), {"files": []})["files"].append(path.name)

        return structure

    def write_to_toml(self, output_file: str) -> None:
        """
        Write the current structure to a TOML file.

        :param output_file: Path to the TOML file.
        """
        with open(output_file, "w") as f:
            toml.dump(self.structure, f)
        logger.info(f"TOML file written to: {output_file}")

    def get_all_paths(self) -> list[Path]:
        """
        Get all full paths to CSV files in the structure.

        :return: A list of Path objects for each CSV file.
        """
        paths = []
        for rel_folder, data in self.structure.items():
            for filename in data.get("files", []):
                full_path = self.root_dir / rel_folder / filename
                paths.append(full_path.resolve())
        return paths

    def find_file(self, name: str) -> list[Path]:
        """
        Find all files in the structure with the given name.

        :param name: File name to search for.
        :return: A list of Path objects to matching files.
        """
        return [p for p in self.get_all_paths() if p.name == name]

    def structure_to_dataframe(self) -> pd.DataFrame:
        """
        Convert self.structure into a DataFrame with columns: ['category', 'subcategory', 'filename'].

        :return: A pandas DataFrame representing the file structure.
        """
        records = []

        for path, values in self.structure.items():
            parts = PurePath(path).parts
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid path format: '{path}'. Expected exactly two components (category/subcategory)."
                )

            category, subcategory = parts

            for filename in values.get("files", []):
                records.append({
                    "category": category,
                    "subcategory": subcategory,
                    "filename": filename
                })

        return pd.DataFrame(records)

    def build_overview_table(self) -> pd.DataFrame:
        """
        Build a boolean overview table.

        - Row labels come from filename (without extension) for permeability/permittivity.
        - Row labels come from subcategory for datasheet curves.
        - Columns are the unique categories.
        - True indicates presence of a file for that label and category.

        :return: dataframe of the overview table
        """
        df = self.structure_to_dataframe()

        records = []

        for _, row in df.iterrows():
            category = row["category"]
            subcategory = row["subcategory"]
            filename = Path(row["filename"]).stem

            if category == "datasheet_curves":
                label = subcategory  # <- label = folder name like "N95"
                col_name = "datasheet_curves"
            else:
                label = filename  # <- label = filename like "N49"
                col_name = f"{category} \n {subcategory}"

            records.append((label, col_name))

        overview_df = pd.DataFrame(records, columns=["label", "column"])

        # Build boolean matrix
        pivot = overview_df.assign(value=True).pivot_table(
            index="label",
            columns="column",
            values="value",
            fill_value=False,
            aggfunc="mean"
        )

        return pivot

    @staticmethod
    def plot_boolean_dataframe(df: pd.DataFrame) -> None:
        """
        Plot a DataFrame with booleans as a colored table: green = True, red = False. Row index is treated as labels.

        :param df: A boolean DataFrame with row index as labels.
        """
        bool_data = df.astype(bool)

        # Convert to correct types for matplotlib.table.Table
        row_labels = df.index.astype(str).tolist()
        col_labels = df.columns.astype(str).tolist()

        cell_text = list(np.where(bool_data.values, "✓", "✗").tolist())
        cell_colors = list(np.where(bool_data.values, "lightgreen", "lightcoral").tolist())

        # Create figure
        fig, ax = plt.subplots(figsize=(0.5 * len(col_labels) + 3, 0.4 * len(row_labels) + 2))

        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        ax.axis('off')
        plt.title("MDB Overview Table", fontsize=14, pad=12)
        plt.tight_layout()
        plt.show()

    def plot_available_data(self, exclude_dc_bias: bool = True) -> None:
        """Plot the existing data of the materialdatabase.

        :param exclude_dc_bias: exclude DC-bias data to prevent an overcrowded plot
        """
        # logger.info(self.build_overview_table())
        # self.plot_boolean_dataframe(self.build_overview_table())
        available_data = self.build_overview_table()

        # exclude_dc_bias:
        if False:
            available_data = available_data[~available_data.index.str.contains("_")]

        logger.info(available_data)
        self.plot_boolean_dataframe(available_data)

    def get_available_h_offset(self, material: Material, data_source: DataSource) -> list[float]:
        """
        Get the list of available h-offsets of the certain material.

        :param material: Material from material database, e.g. mdb.Material.N95
        :type  material: Material
        :param data_source: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  data_source: DataSource
        :return: List of material data
        :rtype:  float
        """
        # Check if requested data is without h-offset
        path2file = Path(f"{self.root_dir}/complex_permeability/{data_source.value}/{material.value}.csv")

        if path2file not in self.all_paths:
            raise ValueError(f"The specified data file with path {path2file} does not exist.")
        else:
            logger.info(f"h-offset read from {path2file}.")
            data_set = pd.read_csv(path2file, sep=",")

            if "h_offset" not in data_set.columns:
                # Create combined dataset
                self.combine_material_permeability_data(material, data_source)
                # Read updated CSV-File again
                data_set = pd.read_csv(path2file, sep=",")
            # Get column with H-offset data
            h_offset_series = data_set['h_offset']
            h_array = h_offset_series.unique()
            # Assemble the result list
            h_offset_list = h_array.tolist()

        return h_offset_list

    def get_complex_data_set(self, material: Material, data_source: DataSource, data_type: ComplexDataType, h_offset: float = 0) -> pd.DataFrame:
        """
        Get a complex data set of a certain material, data type and measurement.

        If the h_offset = 0 (Default parameter) and the data set has no h-offset-value (older format),
        the h-offset will be added to the data. A copy of the old data will be provided.

        :param material: Material from material database, e.g. mdb.Material.N95
        :type  material: Material
        :param data_source: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  data_source: DataSource
        :param data_type: Type of requested data e.g. mdb.ComplexDataType.complex_permeability
        :type  data_type: ComplexDataType
        :param h_offset: H-Offset of the requested data
        :type  h_offset: float
        :return: Requested data within a data frame
        :rtype:  pd.DataFrame
        """
        # Check data type
        if data_type not in {item.value for item in ComplexDataType}:
            raise ValueError(f"{data_type} is no valid complex data type.\n"
                             f"Valid complex data types are: {[item.value for item in ComplexDataType]}")
        else:
            # Check if requested data is without h-offset
            path2file = Path(f"{self.root_dir}/{data_type.value}/{data_source.value}/{material.value}.csv")

            if path2file not in self.all_paths:
                raise ValueError(f"The specified data file with path {path2file} does not exist.")
            else:
                logger.info(f"Complex data read from {path2file}.")
                data_set = pd.read_csv(path2file, sep=",")

                if data_type == ComplexDataType.complex_permeability:
                    if "h_offset" not in data_set.columns:
                        # Create combined dataset
                        self.combine_material_permeability_data(material, data_source)
                        # Read updated CSV-File again
                        data_set = pd.read_csv(path2file, sep=",")
                    # Filter requested H-offset data
                    result_data_set = data_set[data_set['h_offset'] == h_offset]
                    # Check if H-offset dataset is not found
                    if result_data_set.empty:
                        raise ValueError(f"A dataset with h_offset={h_offset} is not available.\n"
                                         f"Please use the 'get_available_h_offset' method to retrieve the list of available h-offsets.")
                else:
                    result_data_set = data_set

                return result_data_set

    def get_complex_permeability(self,
                                 material: Material,
                                 data_source: DataSource,
                                 pv_fit_function: FitFunction,
                                 h_offset: float = 0) -> ComplexPermeability:
        """
        Get a complex permeability data set of a certain material and measurement type.

        :param material: Material from material database, e.g. mdb.Material.N95
        :type  material: Material
        :param data_source: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  data_source: DataSource
        :param pv_fit_function: Algorithm to fit data point by given measurements
        :type  pv_fit_function: FitFunction
        :param h_offset: H-Offset of the requested data
        :type  h_offset: float
        :return: Requested data within a data frame
        :rtype:  pd.DataFrame
        """
        dataset = self.get_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permeability,
            h_offset=h_offset
        )
        return ComplexPermeability(dataset, material, data_source, pv_fit_function)

    def combine_material_permeability_data(self, material: Material, data_source: DataSource) -> bool:
        """
        Combine all files with different h-offset of one material.

        :param material: Material from material database, e.g. mdb.Material.N95
        :type  material: Material
        :param data_source: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  data_source: DataSource
        :return: List of material data
        :rtype:  float
        """
        # Return value
        is_done_successfull = False

        result_list: list[tuple[float, str]]
        df_list: list[pd.DataFrame] = []

        # Assemble path
        data_path = Path(f"{self.root_dir}/complex_permeability/{data_source.value}")
        # Check if data path exists
        if data_path.is_dir():
            # Check for base file
            data_base_file = Path(f"{data_path}/{material.value}.csv")
            # Search for suitable backup name
            data_base_backup_file = Data._get_next_backup_filename(str(data_path), data_base_file.stem + "_backup")
            if data_base_file.is_file():
                # Load base file data
                df_base = pd.read_csv(data_base_file, sep=",")
                df_backup = df_base.copy(deep=True)
                # Search for all h-offset data
                result_list = Data._get_h_dc_offset_data_list(data_path, material)
                # Read and merge dataframes
                for item in result_list:
                    path2file = Path(item[1])
                    df = pd.read_csv(path2file, sep=",")
                    df['h_offset'] = item[0]
                    df_list.append(df)

                # Check if h-offset data frames are found
                if len(df_list):
                    # Check if h_offset is still part of the file
                    if "h_offset" not in df_base.columns:
                        df_base['h_offset'] = 0
                    df_list.append(df_base)
                    # Combine all data frames
                    df_combined = pd.concat(df_list, ignore_index=True)
                    # Remove duplicates
                    df_combined = df_combined.drop_duplicates()
                    # Sort ascending and reset index
                    df_combined = df_combined.sort_values(by=['h_offset', 'T', 'f', 'b'], ascending=[True, True, True, True])

                    # Backup origin file and store new one
                    if not df_combined.equals(df_base):
                        df_backup.to_csv(data_base_backup_file, index=False)
                        df_combined.to_csv(data_base_file, index=False)
                        is_done_successfull = True
                    else:
                        # Notify, that current base file contains all data
                        logger.info(f"File {data_base_file.name} still contains all h-offset data.")
                else:
                    # Notify, that no h-offset file is found
                    logger.info(f"No offset file correspondent to file {data_base_file.name} is found.")
                    # Check if h_offset is still part of the file
                    if "h_offset" not in df_base.columns:
                        # Notify, that actual file has no H-offset column, which is now added
                        logger.info(f"H-offset column is added to file {data_base_file.name}.")
                        df_base['h_offset'] = 0
                        df_backup.to_csv(data_base_backup_file, index=False)
                        df_base.to_csv(data_base_file, index=False)

            else:
                # Log as warning, that no database file exists
                logger.warning(f"{data_base_file.name} does not exist.")
        else:
            # Log as warning about wrong path
            logger.warning(f"Path {data_path.name} does not exist.")

        return is_done_successfull

    @staticmethod
    def _get_next_backup_filename(pathname: str, base_name: str, extension: str = '.csv') -> Path:
        """
        Provide a 'free' file name with a suitable number.

        :param pathname: root directory of material database
        :type  pathname: str
        :param base_name: Material from material database, e.g. mdb.Material.N95
        :type  base_name: Material
        :param extension: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  extension: DataSource
        :return: Path-object with file name, which still not exists in path
        :rtype:  Path
        """
        base_path = pathname + "/" + base_name

        orig_file_path = Path(base_path + extension)
        # Check, if name does not exists
        if not orig_file_path.exists():
            return orig_file_path

        # Search for the next free name (number)
        i = 1
        while True:
            new_name = f"{base_path}{i}{extension}"
            new_file_path = Path(new_name)
            if not new_file_path.exists():
                return new_file_path
            i += 1

    @staticmethod
    def _get_h_dc_offset_data_list(data_path: Path, material: Material) -> list[tuple[float, str]]:
        """
        Get a list of all files with h-dc-offset.

        :param material: Material from material database, e.g. mdb.Material.N95
        :type  material: Material
        :param data_source: Source folder of the material database, e.g. mdb.MeasurementSetup.TDK_MDT
        :type  data_source: DataSource
        :return: List of tuple: ( h offset, path file name)
        :rtype:  list[tuple[float, str]]
        """
        # Variable declaration
        # Number of files in this folder
        number_of_files = 0
        # Result list with  parameter and file names
        result_list: list[tuple[float, str]] = []

        prefix = f"{material.value}"
        # Loop over the files
        for file in data_path.iterdir():
            if file.is_file() and file.stem.startswith(prefix):
                number_of_files += 1
                file_name = str(file.stem)
                # Get the dc-parameter from file name
                h_offset_parameter = Data._get_h_offset_parameter(file_name)
                # Check for valid parameter (h-offset>0)
                if h_offset_parameter != 0:
                    list_item = (h_offset_parameter, str(file))
                    result_list.append(list_item)

        # Evaluate, if minimum 1 file is found
        logger.info(f"{number_of_files} are found.")

        return result_list

    @staticmethod
    def _get_h_offset_parameter(file_name: str) -> float:
        """
        Get a list of all files with h-dc-offset.

        :param file_name: name of the file with h-dc-offset information
        :type  file_name: str
        :return: h dc offset parameter
        :rtype:  float
        """
        # Variable declaration
        parameter_value: float = 0

        # Check if prefix is part of the string (if first letter is a '_' this is to ignore
        start_pos = file_name.find("_", 1)
        if start_pos == -1:
            # No h-dc offset identified
            return parameter_value
        end_pos = file_name.find("Am", start_pos)
        if end_pos == -1:
            # No h-dc offset identified
            return 0
        # Check for only numbers
        try:
            parameter_value = float(file_name[start_pos + 1:end_pos])
        except:
            pass

        return parameter_value

    def get_complex_permittivity(self, material: Material, data_source: DataSource) -> ComplexPermittivity:
        """
        Get a complex permittivity data set of a certain material and measurement type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.LEA_MTB
        :param probe_codes: None -> all probe codes available or select probes via ['Y3F', '7U8'], e.g.
        :return:
        """
        dataset = self.get_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permittivity,
            probe_codes=probe_codes
        )
        return ComplexPermittivity(dataset, material, data_source)

    def set_complex_data_set(self, material: Material, data_source: DataSource, data_type: ComplexDataType, df: pd.DataFrame) -> None:
        """
        Store a complex data set (DataFrame) for a specific material, measurement setup, and data type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        :param data_type: e.g. mdb.ComplexDataType.complex_permeability
        :param df: the DataFrame to store
        """
        if data_type not in {item.value for item in ComplexDataType}:
            raise ValueError(f"{data_type} is not a valid complex data type.\n"
                             f"Valid complex data types are: {[item.value for item in ComplexDataType]}")

        path2file = Path(f"{self.root_dir}/{data_type.name}/{data_source.name}")
        path2file.mkdir(parents=True, exist_ok=True)  # create directories if needed
        file_path = path2file / f"{material.name}.csv"

        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info(f"Complex data written to {file_path}.")

    def set_complex_permeability(self, material: Material, data_source: DataSource, df: pd.DataFrame) -> None:
        """
        Save a complex permeability data set for a given material and measurement type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        :param df: the DataFrame containing the complex permeability data
        """
        self.set_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permeability,
            df=df
        )

    def set_complex_permittivity(self, material: Material, data_source: DataSource, df: pd.DataFrame) -> None:
        """
        Save a complex permittivity data set for a given material and measurement type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.LEA_MTB
        :param df: the DataFrame containing the complex permittivity data
        """
        self.set_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permittivity,
            df=df
        )

    def get_datasheet_curve(self, material: Material, curve_type: DatasheetCurveType) -> pd.DataFrame:
        """
        Get a data sheet curve of a certain material.

        :param material: e.g. mdb.Material.N95
        :param curve_type: e.g. mdb.DatasheetCurveType.mu_amplitude_over_b_at_T
        :return:
        """
        if curve_type not in {item.value for item in DatasheetCurveType}:
            raise ValueError(f"{curve_type} is no valid datasheet curve type.\n"
                             f"Valid curve types are: {[item.value for item in DatasheetCurveType]}")
        else:
            path2file = Path(f"{self.root_dir}/{DatasheetCurvesFolder.name.value}/{material.name}/{curve_type.name}.csv")
            if path2file not in self.all_paths:
                raise ValueError(f"The specified data file with path {path2file} does not exist.")
            else:
                return pd.read_csv(path2file, sep=",")

    def get_datasheet_information(self, material: Material, attribute: DatasheetAttribute) -> Any:
        """
        Get a datasheet attribute value of a certain material.

        :param material: e.g. mdb.Material.N95
        :param attribute: e.g. mdb.DatasheetGeneralInformation.Resistivity
        :return: float value of the requested attribute
        """
        link2info = Path(f"{self.root_dir}/{DatasheetCurvesFolder.name.value}/{material.value}/general_information.csv")
        if link2info not in self.all_paths:
            raise ValueError(f"The specified general information file with path {link2info} does not exist.")

        df_info = pd.read_csv(link2info, sep=",")

        if attribute.value not in df_info.columns:
            raise ValueError(f"Requested attribute '{attribute.value}' not found in {link2info}. "
                             f"Available columns: {list(df_info.columns)}")

        logger.info(f"Attribute {attribute.value} is loaded from material {material.value} datasheet: {df_info[attribute.value].iloc[0]}")
        # Return the scalar value (first row of the requested column)
        return df_info[attribute.value].iloc[0]

    def __str__(self) -> str:
        """
        Return a string representation of the Structure.

        :return: A string showing the root directory path.
        """
        return f"<Structure root={self.root_dir}>"
