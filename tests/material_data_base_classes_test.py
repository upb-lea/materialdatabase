import pytest
import materialdatabase as mdb
import os


def compare_pro_files(first_pro_filepath, second_pro_filepath, significant_digits=6):
    difference = []

    with open(first_pro_filepath, "r") as fd:
        first_content = fd.readlines()

    with open(second_pro_filepath, "r") as fd:
        second_content = fd.readlines()

    for line_count, line in enumerate(first_content):
        if first_content[line_count] == second_content[line_count]:
            pass
        else:
            difference.append(first_content[line_count])
            difference.append(second_content[line_count])
    print(f"{difference = }")


    assert difference == []

@pytest.fixture
def temp_folder():
    # Setup temp folder
    temp_folder_path = os.path.join(os.path.dirname(__file__), "temp")

    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)

    # Test
    yield temp_folder_path

def test(temp_folder):
    database = mdb.MaterialDatabase(is_silent=False)

    T = 100
    f = 100000
    material_name = "N95"
    datasource = "manufacturer_datasheet"
    pro = True
    parent_directory  = temp_folder
    pro_filepath = os.path.join(temp_folder, "core_materials_temp.pro")
    pro_verification_filepath = os.path.join(os.path.dirname(__file__), "fixtures", "core_materials_temp_n95_100000Hz_100deg.pro")

    database.permeability_data_to_pro_file(T, f, material_name, datasource, pro, parent_directory)

    compare_pro_files(pro_filepath, pro_verification_filepath)
