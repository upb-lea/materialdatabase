import material_data_base_classes as mdb

database = mdb.MaterialDatabase()

# -----Enter the freq and Temp-----------
database.permeability_data_to_pro_file(T=55, f=100000, material_name="N87", datasource="manufacturer_datasheet", pro=True)

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
# mdb.compare_core_loss_flux_density_data(material_list=["N95", "N87"], temperature=None)
# mdb.compare_core_loss_temperature(material_list=["N95", "N87"], flux=None)
# mdb.compare_core_loss_frequency(material_list=["N95", "N87"], temperature=100)
# mdb.compare_b_h_curve(material_list=["N95", "N87"], temperature=None)
# database.get_material_property(material_name="N87", property="volumetric_mass_density")
# mdb.compare_permeability_measurement_data(material_list=["N87", "N95"], frequency=200000)
