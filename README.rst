Material database for power electronic usage
===============================================

The materialdatabase mainly contains complex magnetic and dielectric data of soft-magnetic ferrite materials under sinusoidal excitation. Typical target applications of the provided material data are magentic components in power electronic converters operated at switching frequencies reaching from 100 kHz to 3 MHz. The database allows to compare materials and measurement setups, to work with fit functions (e.g., Steinmetz), and features data export as interpolation grids. FEM solvers like Comsol can directly import these interpolation grids and incorporate the data in frequency-domain simulations. The amplitude-dependency of the complex permeability is then typically modelled via local linearization (e.g., based on Newton iterations). The open-source FEM Magnetics Toolbox (`FEMMT <https://github.com/upb-lea/FEM_Magnetics_Toolbox>`__) automatically allows to import material data from this database.

Overview
-------------------

..
    Key features
    ~~~~~~~~~~~~~~~~~
..

* Measurement data is stored in a .csv-files for each combination of material and measurement setup.
* Data sources:
    * Complex Permeability:
        * Datasheet:
            Data based on datasheet or manufacturer design tool data (e.g., `TDK MDT <https://www.tdk-electronics.tdk.com/de/194550/design-support/tools-fuer-entwickler/ferrite-magnetic>`__ ). The complex permeability is computed from the given loss densit< and amplitude permeability.
        * MagNet:
            Data from the `MagNet <https://mag-net.princeton.edu/>`__ project. The time series data of the sinusoidal measurements is simplified to complex permeability assuming perfectly elliptical hysteresis. For the MagNet data, also DC bias is provided.
        * LEA MTB (LEA Material Test Bench):
            Data from measurements at the department of Power Electronics and Electrical Drives at Paderborn University. The measurements are taken with a setup employing the `capacitive compensated two-winding method <https://ieeexplore.ieee.org/document/6648460>`__.
    * Complex Permittivity:
        * LEA MTB (LEA Material Test Bench):
            Measurements at the department of Power Electronics and Electrical Drives at Paderborn University. The dielectric measurements are taken with a Wayne Kerr 6500b impedance analyzer on thin silver-plated cuboidal cores.
* The database currently contains the ferrite materials listed in the following table:

|material_overview|

Installation
---------------

::

    pip install materialdatabase

Example Data
------------------------------------
The data is stored in .csv files:

|material_data|


Example Code
------------------------------------

Material properties can be loaded as follows:
::
    mdb_data = mdb.Data()
    material_name = mdb.Material.N95

    permeability = mdb_data.get_complex_permeability(material=material_name,
                                                     data_source=mdb.DataSource.LEA_MTB,
                                                     pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
    print(f"Exemplary complex permeability data: \n {permeability.measurement_data.head()} \n")

    permittivity = mdb_data.get_complex_permittivity(material=material_name,
                                                     data_source=mdb.DataSource.LEA_MTB)
    print(f"Exemplary complex permittivity data: \n {permittivity.measurement_data.head()} \n ")


Output:
::
    Exemplary complex permeability data:
                probe        f       T         b      mu_real     mu_imag  h_offset
    0  R29.5x19x14.9  59629.0  30.145  0.040125  3955.516035  333.486173         0
    1  R29.5x19x14.9  58929.0  30.154  0.049834  4050.789996  380.241871         0
    2  R29.5x19x14.9  58329.0  30.154  0.058502  4138.476021  424.009115         0
    3  R29.5x19x14.9  57929.0  30.147  0.066572  4239.539939  458.079275         0
    4  R29.5x19x14.9  57529.0  30.158  0.074332  4303.405675  488.755116         0

    Exemplary complex permittivity data:
       probe         f   T      eps_real      eps_imag
    0   LE2  107760.0  28  95567.710843  36977.344836
    1   LE2  121546.0  28  93791.671181  35002.540113
    2   LE2  137096.0  28  92085.767038  33189.049457
    3   LE2  154635.0  28  90442.389511  31563.885192
    4   LE2  174418.0  28  88863.933437  30088.808585


Detailed examples with material comparisons and data exporting can be found in the "examples" folder.




Usage via FEM Magnetics Toolbox (FEMMT)
-------------------

`FEMMT <https://github.com/upb-lea/FEM_Magnetics_Toolbox>`__ can be installed using the python pip package manager.

::

    pip install femmt

For working with the latest version, refer to the `documentation <https://upb-lea.github.io/FEM_Magnetics_Toolbox/intro.html>`__


Bug Reports
--------------

Please use the issues report button within github to report bugs.


Changelog
------------

Find the changelog `here <CHANGELOG.md>`__.

.. |material_overview| image:: /docs/source/figures/overview.jpg
.. |material_data| image:: /docs/source/figures/exemplary_N95_permeability_data.jpg
