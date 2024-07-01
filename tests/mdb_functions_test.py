"""Provides some functions to test/check other functions."""
import pytest
import materialdatabase as mdb
import numpy as np
import deepdiff

mdb_instance = mdb.MaterialDatabase()


def test_get_material_attribute():
    """
    Tests the functionality to get the material properties of a ferrite of the material database.

    :return: None
    """
    initial_mu_r_abs = mdb_instance.get_material_attribute(material_name=mdb.Material.N95, attribute="initial_permeability")
    assert initial_mu_r_abs == pytest.approx(3000, rel=1e-3)

    core_material_resistivity = mdb_instance.get_material_attribute(material_name=mdb.Material.N95, attribute="resistivity")
    assert core_material_resistivity == pytest.approx(6, rel=1e-3)

    b_ref, mu_r_real, mu_r_imag = mdb_instance.permeability_data_to_pro_file(temperature=25, frequency=150000, material_name=mdb.Material.N95,
                                                                             datatype=mdb.MeasurementDataType.ComplexPermeability,
                                                                             datasource=mdb.MaterialDataSource.ManufacturerDatasheet, parent_directory="")

    b_ref = np.array(b_ref)

    b_test_ref = np.array([0.0, 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.0])
    mu_r_real_test_ref = np.array([1., 438.42857143, 550.71428571, 560., 560., 560., 560., 560.])
    mu_r_imag_test_ref = np.array([3.00000000e+03, 2.96707143e+03, 2.95635714e+03, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00])

    assert b_ref == pytest.approx(b_test_ref, rel=1e-3)
    assert mu_r_real == pytest.approx(mu_r_real_test_ref, rel=1e-3)
    assert mu_r_imag == pytest.approx(mu_r_imag_test_ref, rel=1e-3)

    epsilon_r, epsilon_phi_deg = mdb_instance.get_permittivity(temperature=25, frequency=150000, datasource=mdb.MaterialDataSource.Measurement,
                                                               material_name=mdb.Material.N95, datatype=mdb.MeasurementDataType.ComplexPermittivity,
                                                               measurement_setup=mdb.MeasurementSetup.LEA_LK, interpolation_type="linear")
    assert epsilon_r == pytest.approx(89591, rel=1e-3)
    assert epsilon_phi_deg == pytest.approx(19.6, rel=1e-3)


def test_interpolation():
    """
    Tests the function interpolate_b_dependent_quantity_in_temperature_and_frequency().

    :return: None
    """
    temperature = 45
    frequency = 140000
    T_low = 25
    T_high = 80
    f_low = 100000
    f_high = 200000

    b_T_low_f_low = [1, 2, 3, 4]
    mu_T_low_f_low = [1000, 2000, 3000, 4000]

    b_T_high_f_low = [2, 3, 4, 5]
    mu_T_high_f_low = [1300, 1500, 1700, 1750]

    b_T_low_f_high = [1.1, 2, 3, 4]
    mu_T_low_f_high = [4000, 5000, 6000, 7000]

    b_T_high_f_high = [1, 2.2, 3, 4.5]
    mu_T_high_f_high = [2300, 2500, 3000, 3200]

    (result) = mdb.interpolate_b_dependent_quantity_in_temperature_and_frequency(temperature, frequency, T_low, T_high, f_low, f_high,
                                                                                 b_T_low_f_low, mu_T_low_f_low, b_T_high_f_low, mu_T_high_f_low,
                                                                                 b_T_low_f_high, mu_T_low_f_high, b_T_high_f_high, mu_T_high_f_high)

    result = np.array(result)
    correct_result = [[2.0, 2.28571429, 2.57142857, 2.85714286, 3.14285714, 3.42857143, 3.71428571, 4.0],
                      [2678.78787879, 2885.71428571, 3105.97402597, 3326.23376623, 3536.27705628, 3736.1038961, 3935.93073593, 4135.75757576]]
    assert result[0] == pytest.approx(correct_result[0], rel=1e-3)
    assert result[1] == pytest.approx(correct_result[1], rel=1e-3)


def test_neighbourhood():
    """
    Tests the class neighbourhood.

    :return: None
    """
    correct_result_neighbourhood = {'T_low_f_low': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 300000, 'index': 2},
                                    'epsilon_r': 40993.333333333336, 'epsilon_phi_deg': 24.637},
                                    'T_low_f_high': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 400000, 'index': 3},
                                                     'epsilon_r': 38544.333333333336, 'epsilon_phi_deg': 22.727},
                                    'T_high_f_low': {'temperature': {'value': 100, 'index': 1}, 'frequency': {'value': 300000, 'index': 2},
                                                     'epsilon_r': 45899.333333333336, 'epsilon_phi_deg': 28.648},
                                    'T_high_f_high': {'temperature': {'value': 100, 'index': 1}, 'frequency': {'value': 400000, 'index': 3},
                                                      'epsilon_r': 42960.666666666664, 'epsilon_phi_deg': 27.502}}

    list_of_permittivity_dicts = mdb_instance.load_permittivity_measurement(material_name="N49", datasource="measurements", measurement_setup="LEA_LK")

    T = 64
    f = 450000

    neighbourhood = mdb.create_permittivity_neighbourhood(temperature=T, frequency=f, list_of_permittivity_dicts=list_of_permittivity_dicts)

    difference = deepdiff.DeepDiff(neighbourhood, correct_result_neighbourhood, ignore_order=True, significant_digits=3)
    print(f"{difference=}")

    assert not deepdiff.DeepDiff(neighbourhood, correct_result_neighbourhood, ignore_order=True, significant_digits=3)


def test_load_permittivity_measurement():
    """
    Tests the function load_permittivity_measurement().

    :return: None
    """
    load_permittivity_measurement_result = {'T_low_f_low': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 100000.0, 'index': 0},
                                                            'epsilon_r': 61294.333333333336, 'epsilon_phi_deg': 36.85999999999999},
                                            'T_low_f_high': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 100000.0, 'index': 0},
                                                             'epsilon_r': 61294.333333333336, 'epsilon_phi_deg': 36.85999999999999},
                                            'T_high_f_low': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 100000.0, 'index': 0},
                                                             'epsilon_r': 61294.333333333336, 'epsilon_phi_deg': 36.85999999999999},
                                            'T_high_f_high': {'temperature': {'value': 60, 'index': 0}, 'frequency': {'value': 100000.0, 'index': 0},
                                                              'epsilon_r': 61294.333333333336, 'epsilon_phi_deg': 36.85999999999999}}

    list_of_permittivity_dicts = mdb_instance.load_permittivity_measurement(material_name="N49", datasource="measurements", measurement_setup="LEA_LK")

    create_dict = mdb.create_permittivity_neighbourhood(temperature=60, frequency=1e5, list_of_permittivity_dicts=list_of_permittivity_dicts)

    assert not deepdiff.DeepDiff(load_permittivity_measurement_result, create_dict, ignore_order=True, significant_digits=3)


def test_mypolate():
    """
    Tests the function my_polate_linear().

    :return: None
    """
    a = 10
    b = 20
    f_a = 100
    f_b = 200

    # interpolate
    x = 15
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 150.0

    # low extrapolate
    x = 5
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 50.0

    # negative low extrapolate
    x = -5
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == -50.0

    # high extrapolate
    x = 25
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 250.0

    # high exact: x=b
    x = b
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 200.0

    # low exact: x=a
    x = a
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 100.0

    a = 10
    b = 10
    f_a = 100
    f_b = 100

    # interpolate
    x = 10
    result = mdb.my_polate_linear(a, b, f_a, f_b, x)
    assert result == 100


def test_find_nearest_neighbours():
    """
    Tests the function find_nearest_neighbours().

    :return: None
    """
    list_to_search_in = [10]

    print("Case 0")
    value = "computer"
    with pytest.raises(TypeError):
        mdb.find_nearest_neighbours(value, list_to_search_in)

    list_to_search_in = [10, 20, 30, 40, 60]

    print("Case 1")
    value = 10.0
    result = mdb.find_nearest_neighbours(value, list_to_search_in)
    assert (0, 10, 0, 10) == result

    print("Case 2")
    value = 25
    result = mdb.find_nearest_neighbours(value, list_to_search_in)
    assert (1, 20, 2, 30) == result

    print("Case 3a")
    value = 5
    result = mdb.find_nearest_neighbours(value, list_to_search_in)
    assert (0, 10, 1, 20) == result

    print("Case 3b")
    value = 80
    result = mdb.find_nearest_neighbours(value, list_to_search_in)
    assert (3, 40, 4, 60) == result
