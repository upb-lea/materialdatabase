import material_data_base_classes as m

database = m.MaterialDatabase()

# -----Enter the freq and Temp-----------
database.get_data_at_working_point(T=60, f=160000, material_name="N95")

# ------material properties to be plotted-----
# database.plot_data(material_name="N95", properties="mu_real")
# database.plot_data(material_name="N95", properties="mu_imag")
# database.plot_data(material_name="N95", properties="B_H curve")
# -------Enter the file format to export the data-----
# database.export_data(format="pro")
