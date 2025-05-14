import os
from pathlib import Path
import toml
from materialdatabase.meta.data_enums import *

class Data:
    def __init__(self, root_dir: str):
        """
        Initialize Structure by scanning a folder structure.

        Args:
            root_dir (str): The directory to scan for CSV files.
        """
        self.root_dir = Path(root_dir).resolve()
        self.structure = self._scan_structure()

    def _scan_structure(self):
        """
        Recursively scans the folder for CSV files and builds the structure dict.

        Returns:
            dict: A mapping of relative folder paths to CSV file lists.
        """
        structure = {}

        for dirpath, _, filenames in os.walk(self.root_dir):
            csv_files = [f for f in filenames if f.lower().endswith('.csv')]
            if csv_files:
                rel_path = os.path.relpath(dirpath, self.root_dir)
                structure[rel_path] = {"files": csv_files}

        return structure

    def write_to_toml(self, output_file: str):
        """
        Writes the current folder structure to a TOML file.

        Args:
            output_file (str): Path to the TOML file.
        """
        with open(output_file, "w") as f:
            toml.dump(self.structure, f)
        print(f"TOML file written to: {output_file}")

    def get_all_paths(self):
        """
        Return full paths to all CSV files found in the structure.

        Returns:
            List[Path]: Paths to all CSV files.
        """
        paths = []
        for rel_folder, data in self.structure.items():
            for filename in data.get("files", []):
                full_path = self.root_dir / rel_folder / filename
                paths.append(full_path.resolve())
        return paths

    def find_file(self, name: str):
        """
        Find all files with the given name.

        Args:
            name (str): File name to search for.

        Returns:
            List[Path]: Matching file paths.
        """
        return [p for p in self.get_all_paths() if p.name == name]


    def __str__(self):
        return f"<Structure root={self.root_dir}>"