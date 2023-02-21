import pytest
import materialdatabase as mdb
import os
import numpy as np
material_db = mdb.MaterialDatabase()

def test_get_material_property():

    initial_mu_r_abs = material_db.get_material_property(material_name="N95", property="initial_permeability")
    assert initial_mu_r_abs == pytest.approx(3000, rel=1e-3)

    core_material_resistivity = material_db.get_material_property(material_name="N95", property="resistivity")
    assert core_material_resistivity == pytest.approx(6, rel=1e-3)


    b_ref, mu_r_real, mu_r_imag = material_db.permeability_data_to_pro_file(temperature=25, frequency=150000, material_name = "N95", datatype = "complex_permeability",
                                      datasource = mdb.MaterialDataSource.ManufacturerDatasheet, parent_directory = "")

    b_ref = np.array(b_ref)

    b_test_ref = np.array([0.0 , 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.0])
    mu_r_real_test_ref = np.array([  1., 438.42857143, 550.71428571, 560., 560., 560., 560., 560.])
    mu_r_imag_test_ref = np.array([3.00000000e+03, 2.96707143e+03, 2.95635714e+03, 1.00000000e+00,
       1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00])

    assert b_ref == pytest.approx(b_test_ref, rel=1e-3)
    assert mu_r_real == pytest.approx(mu_r_real_test_ref, rel=1e-3)
    assert mu_r_imag == pytest.approx(mu_r_imag_test_ref, rel=1e-3)

    epsilon_r, epsilon_phi_deg = material_db.get_permittivity(temperature= 25, frequency=150000, material_name = "N95", datasource = "measurements",
                                      datatype = mdb.MeasurementDataType.ComplexPermittivity, measurement_setup = "LEA_LK",interpolation_type = "linear")
    assert epsilon_r == pytest.approx(89591, rel=1e-3)
    assert epsilon_phi_deg == pytest.approx(19.6, rel=1e-3)
