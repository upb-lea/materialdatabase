"""Functions to load permittivity data from the LEA Magnetics Testbench in the material database."""

import logging

import os
from typing import Dict, List, Any

import numpy as np
from scipy import constants
import pandas as pd

logger = logging.getLogger(__name__)


def collect_probes_and_temperatures(root_dir: str = "raw") -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Collect available probes and their measurement temperatures from a directory structure.

    :param root_dir: Root directory containing material folders
    :type root_dir: str
    :return: Nested dictionary of materials, probes, and their temperature measurements
    :rtype: Dict[str, Dict[str, Dict[str, List[int]]]]
    """
    results: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    # loop over materials
    for material in os.listdir(root_dir):
        material_path = os.path.join(root_dir, material)
        if not os.path.isdir(material_path):
            continue

        results[material] = {}

        # loop over probes
        for probe in os.listdir(material_path):
            probe_path = os.path.join(material_path, probe)
            if not os.path.isdir(probe_path):
                continue

            temperatures: List[int] = []
            for file in os.listdir(probe_path):
                if file.endswith(".csv") and file != "probe_dimensions.csv":
                    temp_name = os.path.splitext(file)[0]  # remove ".csv"
                    temperatures.append(int(temp_name))

            results[material][probe] = {
                "temperatures": sorted(temperatures)
            }

            print(f"Material: {material}, Probe: {probe}, Temperatures: {results[material][probe]['temperatures']}")

    return results
