"""Provides some functions to test/check classes."""
import pytest
import materialdatabase as mdb
import os


def compare_pro_files(first_pro_filepath, second_pro_filepath, significant_digits=6):
    """
    Compare two files of possible differences.

    If differences exist -> function raises AssertionError
    :param first_pro_filepath: path of first file
    :param second_pro_filepath: path of second file
    :param significant_digits: number of significant digits
    :return: None
    """
    difference = []

    with open(first_pro_filepath, "r") as fd:
        first_content = fd.readlines()

    with open(second_pro_filepath, "r") as fd:
        second_content = fd.readlines()

    for line_count, _ in enumerate(first_content):
        if first_content[line_count] == second_content[line_count]:
            pass
        else:
            difference.append(first_content[line_count])
            difference.append(second_content[line_count])
    print(f"{difference=}")

    assert difference == []


@pytest.fixture
def temp_folder():
    """
    Create a folder with name temp in same path as this file.

    :return: None
    """
    # Setup temp folder
    temp_folder_path = os.path.join(os.path.dirname(__file__), "temp")

    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)

    # Test
    yield temp_folder_path


def test(temp_folder):
    """
    Tests the path of the created temporary folder.

    :param temp_folder: path to folder temp
    :return: None
    """
    database = mdb.MaterialDatabase(is_silent=False)

    T = 100
    f = 100000
    material_name = mdb.Material.N95
    datasource = mdb.MaterialDataSource.ManufacturerDatasheet
    parent_directory = temp_folder
    pro_filepath = os.path.join(temp_folder, "core_materials_temp.pro")
    pro_verification_filepath = os.path.join(os.path.dirname(__file__), "fixtures", "core_materials_temp_n95_100000Hz_100deg.pro")
    b_ref, mu_r_imag, mu_r_real = database.permeability_data_to_pro_file(temperature=T, frequency=f, material_name=material_name, datasource=datasource,
                                                                         datatype="complex_permeability", parent_directory=parent_directory)
    print(f"{b_ref=}")
    print(f"{mu_r_imag=}")
    print(f"{mu_r_real=}")

    compare_pro_files(pro_filepath, pro_verification_filepath)
