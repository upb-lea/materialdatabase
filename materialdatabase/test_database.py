import material_data_base_classes as mdb

database = mdb.MaterialDatabase()

# -----Enter the freq and Temp-----------
# database.permeability_data_to_pro_file(T=30, f=300000, material_name="N95", datasource="measurements", pro=True)

# ------material properties to be plotted-----
# database.plot_data(material_name="N95", properties="mu_real")
# database.plot_data(material_name="N95", properties="mu_imag")
# database.plot_data(material_name="N95", properties="b_h_curve")
# -------Enter the file format to export the data-----
# database.export_data(file_format="pro")
# database.get_steinmetz_data(material_name="N95", type="Steinmetz", datasource="measurements")
# database.get_initial_permeability(material_name="N95")
# database.get_resistivity(material_name="N95")
# --------------compare-----------
# mdb.compare_core_loss_flux_density_data(material_list=["N95", "N87"], temperature_list = [ 25, 100])
# mdb.compare_core_loss_temperature(material_list=["N49", "N87"], flux_list=[0.1, 0.1])
# mdb.compare_core_loss_frequency(material_list=["N95", "N87"], temperature_list=[25, 100], flux_list=[0.1, 0.1] )
# mdb.compare_b_h_curve(material_list=["N95", "N87"], temperature=None)
# database.get_material_property(material_name="N49", property="volumetric_mass_density")
# mdb.compare_permeability_measurement_data(material_list=["N87", "N95"], frequency_list=[300000,300000], temperature_list=[30,30],plot_real_part=True)
# print(mdb.drop_down_list(material_name="N49", temperature=False, flux=False, freq= True, comparison_type="mvm"))
# mdb.material_list_in_database(material_list=True)
mdb.compare_core_loss_flux_datasheet_measurement(material="N49", temperature_list = [ 25, 60])