"""Example file to show how to process data with the ComplexPermeability class."""
import materialdatabase as mdb
import logging
import numpy as np
from matplotlib import pyplot as plt

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Define an opration point
b_common = np.linspace(0, 0.2, 20)
f_op = 950_000
T_op = 65

# init a material database instance
mdb_data = mdb.Data()

# load ComplexMaterial instance
mu_N49 = mdb_data.get_complex_permeability(material=mdb.Material.N49,
                                           data_source=mdb.DataSource.TDK_MDT,
                                           pv_fit_function=mdb.FitFunction.enhancedSteinmetz)

# Fit operation point
mu_real_op, mu_imag_op = mu_N49.fit_real_and_imaginary_part_at_f_and_T(
    f_op=f_op,
    T_op=T_op,
    b_vals=b_common
)

plt.plot(b_common, mu_real_op)
plt.plot(b_common, mu_imag_op)
plt.show()
