"""Examples of differtent functions for the material database."""
import material_data_base_classes as mdb

database = mdb.MaterialDatabase()

# -----Enter the freq and Temp-----------
# database.permeability_data_to_pro_file(temperature=30, frequency=300000, material_name="N95", datasource="measurements", pro=True)

# ------material properties to be plotted-----
# database.plot_data(material_name="N95", properties="mu_real")
# database.plot_data(material_name="N95", properties="mu_imag")
# database.plot_data(material_name="N95", properties="b_h_curve")
# -------Enter the file format to export the data-----
# database.export_data(file_format="pro")
# database.get_steinmetz_data(material_name="N95", type="Steinmetz", datasource="measurements")
# database.get_material_property(material_name="N49", property="initial_permeability")

# --------------compare-----------
# database.compare_core_loss_flux_density_data(material_list=["N95", "PC200"], temperature_list = [ 25, 25])
# database.compare_core_loss_temperature(material_list=["N49"], flux_density_list=[0.1])
# database.compare_core_loss_frequency(material_list=["N49", "N87"], temperature_list=[25, 100], flux_density_list=[0.1, 0.1] )
# database.compare_b_h_curve(material_list=["N95", "N87"], temperature_list=[ 25, 25])
# database.get_material_property(material_name="N49", property="volumetric_mass_density")
# database.compare_permeability_measurement_data(material_list=["N87", "N95"],measurement_name=['LEA_LK','LEA_LK'],
# frequency_list=[300000,300000], temperature_list=[30,30],plot_real_part=True)
# print(database.drop_down_list(material_name="N49",datatype=None, measurement_name=None ,temperature=False,
# flux_density=True, frequency= False, comparison_type="dvd"))
# print(database.material_list_in_database())
# database.compare_core_loss_flux_datasheet_measurement(material="N49", measurement_name='LEA_LK', temperature_list = [ 25, 60])
# print(database.find_measurement_names(material_name="N95", datatype="complex_permeability"))
