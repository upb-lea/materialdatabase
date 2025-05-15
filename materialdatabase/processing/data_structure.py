"""Class to represent the data structure and load material data."""
import os
from pathlib import Path
import toml
from materialdatabase.meta.data_enums import *
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class Data:
    """Represent the structure of a folder tree containing CSV files."""

    def __init__(self, root_dir: str):
        """
        Initialize the Structure by scanning a directory.

        :param root_dir: The base directory to scan for CSV files.
        """
        self.root_dir = Path(root_dir).resolve()
        self.structure = self._scan_structure()
        self.all_paths = self.get_all_paths()

    def _scan_structure(self) -> dict:
        """
        Scan the folder tree for CSV files.

        :return: A dictionary mapping relative folder paths to lists of CSV files.
        """
        structure = {}

        for dirpath, _, filenames in os.walk(self.root_dir):
            csv_files = [f for f in filenames if f.lower().endswith('.csv')]
            if csv_files:
                rel_path = os.path.relpath(dirpath, self.root_dir)
                structure[rel_path] = {"files": csv_files}

        return structure

    def write_to_toml(self, output_file: str) -> None:
        """
        Write the current structure to a TOML file.

        :param output_file: Path to the TOML file.
        """
        with open(output_file, "w") as f:
            toml.dump(self.structure, f)
        print(f"TOML file written to: {output_file}")

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
            try:
                category, subcategory = path.split("\\")
            except ValueError as err:
                raise ValueError(
                    f"Invalid path format: '{path}'. Expected 'category\\subcategory'."
                ) from err  # <-- keep original traceback

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
            category = row['category']
            subcategory = row['subcategory']
            filename = row['filename']

            # Determine row label based on category
            if category == "datasheet_curves":
                label = subcategory
            else:
                label = os.path.splitext(filename)[0]

            records.append((label, category))

        # Build DataFrame from label-category pairs
        overview_df = pd.DataFrame(records, columns=["label", "category"])

        # Add a 'value' column of True, then pivot using 'any' to stay in boolean domain
        pivot = overview_df.assign(value=True).pivot_table(
            index="label",
            columns="category",
            values="value",
            fill_value=False,
            aggfunc='any'  # <-- This safely resolves ambiguity
        )

        return pivot

    @staticmethod
    def plot_boolean_dataframe(df: pd.DataFrame) -> None:
        """
        Plot a DataFrame with booleans as a colored table: green = True, red = False. Row index is treated as labels.

        :param df:
        """
        fig, ax = plt.subplots(figsize=(len(df.columns) + 2, len(df.index) + 1))

        row_labels = df.index.astype(str)
        bool_data = df.astype(bool)

        # Build color grid
        colors = np.where(bool_data.values, 'lightgreen', 'lightcoral')

        # Draw table
        table = plt.table(
            cellText=None,
            rowLabels=row_labels,
            colLabels=df.columns,
            cellColours=colors,
            loc='center',
            cellLoc='center'
        )

        table.scale(1.2, 1.5)
        ax.axis('off')
        plt.title("MDB Overview Table", pad=20)
        plt.tight_layout()
        plt.show()

    def plot_available_data(self) -> None:
        """Plot the existing data of the materialdatabase."""
        self.plot_boolean_dataframe(self.build_overview_table())

    def get_complex_data_set(self, material: Material, measurement_setup: MeasurementSetup, data_type: ComplexDataType) -> pd.DataFrame:
        """
        Get a complex data set of a certain material, data type and measurement.

        :param material:
        :param measurement_setup:
        :param data_type:
        :return:
        """
        if data_type not in {item.value for item in ComplexDataType}:
            raise ValueError(f"{data_type} is no valid complex data type.\n"
                             f"Valid complex data types are: {[item.value for item in ComplexDataType]}")
        else:
            path2file = Path(f"{self.root_dir}/{data_type.name}/{measurement_setup.name}/{material.name}.csv")
            if path2file not in self.all_paths:
                raise ValueError(f"The specified data file with path {path2file} does not exist.")
            else:
                return pd.read_csv(path2file, sep=",")

    def get_datasheet_curve(self, material: Material, curve_type: DatasheetCurveType) -> pd.DataFrame:
        """
        Get a data sheet curve of a certain material.

        :param material:
        :param curve_type:
        :return:
        """
        if curve_type not in {item.value for item in DatasheetCurveType}:
            raise ValueError(f"{curve_type} is no valid datasheet curve type.\n"
                             f"Valid curve types are: {[item.value for item in DatasheetCurveType]}")
        else:
            path2file = Path(f"{self.root_dir}/{DatasheetCurvesFolder.name.name}/{material.name}/{curve_type}.csv")
            if path2file not in self.all_paths:
                raise ValueError(f"The specified data file with path {path2file} does not exist.")
            else:
                return pd.read_csv(path2file, sep=",")

    def __str__(self) -> str:
        """
        Return a string representation of the Structure.

        :return: A string showing the root directory path.
        """
        return f"<Structure root={self.root_dir}>"
