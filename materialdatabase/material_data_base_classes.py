

class MaterialDatabase:
    """
    This class manages the data stored in the material database.
    It has the possibility to export for example soft magnetic materials' loss data in certain format,
    so that it can easily be interfaced by tools like the FEM Magnetic Toolbox.
    """
    def __init__(self):
        print("The material database is now initialized")

    def get_data_at_working_point(self, T, f, material_name):
        """
        Method is used to read data from the material database.
        :param T:
        :param f:
        :param material_name:
        :return:
        """
        print(f"Material properties of {material_name} are loaded at {T} °C and {f} Hz.")
        pass

    def export_data(self, data_to_export, format):
        """
        Method is used to export data from the material database in a certain file format.
        :param data_to_export:
        :param format:
        :return:
        """
        print(f"Data {data_to_export} is exported in a {format}-file.")
        pass

    def store_data(self, material_name, data_to_be_stored):
        """
        Method is used to store data from measurement/datasheet into the material database.
        :param material_name:
        :param data_to_be_stored:
        :return:
        """
        print(f"Material properties of {material_name} are stored in the material database.")
        pass

    def plot_data(self, material_name, properties):
        """
        Method is used to plot certain material properties of materials.
        :param properties:
        :param material_name:
        :return:
        """
        print(f"Material properties {properties} of {material_name} are plotted.")
        pass

    def compare_data(self, material_name_list):
        """
        Method is used to compare material properties.
        :param material_name_list:
        :return:
        """
        print(f"Material properties of {material_name_list} are compared.")
        pass

