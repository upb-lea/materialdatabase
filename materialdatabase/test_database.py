import material_data_base_classes as m

database = m.MaterialDatabase()

# -----Enter the freq and Temp-----------
# database.get_permeability_data(T=50, f=150000, material_name="N95", datasource="measurements")

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
m.compare_core_loss_flux_density_data(material_list=["N95", "N87"], temperature=25)
m.compare_core_loss_temperature(material_list=["N95", "N87"], flux=200e-3)
