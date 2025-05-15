"""Example file to show how to plot a overview of the existing material data."""
import materialdatabase as mdb

# Init the materialdatabase
mdb_data = mdb.Data(root_dir=mdb.get_user_paths().material_data)

# Plot the material overview in a colored table
mdb_data.plot_available_data()
