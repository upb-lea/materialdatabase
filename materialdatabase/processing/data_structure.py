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
        available_data = self.build_overview_table()

        if exclude_dc_bias:
            available_data = available_data[~available_data.index.str.contains("_")]

        logger.info(available_data)
        self.plot_boolean_dataframe(available_data)

    def get_complex_data_set(self, material: Material, data_source: DataSource, data_type: ComplexDataType) -> pd.DataFrame:
        """
        Get a complex data set of a certain material, data type and measurement.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        :param data_type: e.g. mdb.ComplexDataType.complex_permeability
        :return:
        """
        if data_type not in {item.value for item in ComplexDataType}:
            raise ValueError(f"{data_type} is no valid complex data type.\n"
                             f"Valid complex data types are: {[item.value for item in ComplexDataType]}")
        else:
            path2file = Path(f"{self.root_dir}/{data_type.value}/{data_source.value}/{material.value}.csv")
            if path2file not in self.all_paths:
                raise ValueError(f"The specified data file with path {path2file} does not exist.")
            else:
                logger.info(f"Complex data read from {path2file}.")
                return pd.read_csv(path2file, sep=",")

    def get_complex_permeability(self,
                                 material: Material,
                                 data_source: DataSource,
                                 pv_fit_function: FitFunction) -> ComplexPermeability:
        """
        Get a complex permeability data set of a certain material and measurement type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.TDK_MDT
        :param pv_fit_function:
        :return:
        """
        dataset = self.get_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permeability
        )
        return ComplexPermeability(dataset, material, data_source, pv_fit_function)

    def get_complex_permittivity(self, material: Material, data_source: DataSource) -> ComplexPermittivity:
        """
        Get a complex permittivity data set of a certain material and measurement type.

        :param material: e.g. mdb.Material.N95
        :param data_source: e.g. mdb.MeasurementSetup.LEA_MTB
        :return:
        """
        dataset = self.get_complex_data_set(
            material=material,
            data_source=data_source,
            data_type=ComplexDataType.complex_permittivity
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
