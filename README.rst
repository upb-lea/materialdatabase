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
            Measurements at the department of Power Electronics and Electrical Drives at Paderborn University. The dielectric measurements are taken with thin silver-plated cuboidal cores.
* The database currently contains the ferrite materials listed in the following table:

|material_overview|

..
    * Input features:
        * Write magnetic parameters into the database
            * Amplitude of permeability
            * Angle of permeability
            * Power loss density (hysteresis losses)
            * Magnetic flux density
            * Magnetic field strength

        * Write electric parameters into the database
            * Amplitude of permittivity
            * Angle of permittivity
            * Power loss density (eddy current losses)
            * Electric flux density
            * Electric field strength

        * Write datasheet data into the database

    * Output features:
        * Get the magnetic parameters from the database
        * Providing permeability and permittivity data for `FEMMT <https://github.com/upb-lea/FEM_Magnetics_Toolbox>`__

    * Interpolation of material data (both electric and magnetic parameters)

    * GUI features (included in `FEMMT <https://github.com/upb-lea/FEM_Magnetics_Toolbox>`__):
        * Compare the datasheet values of different ferrite cores (e.g. BH-curves or power-loss curves)
        * Materials for comparison:
            * N95
            * N87
            * N49
            * PC200
            * DMR96A
..

..
    Planned features (Roadmap for 202x)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * Input features:
        * Universal function to write data into the database

    * Output features:
        * Get the electric parameters from the database
        * Extract data from the database as a specific data file (e.g. .csv)

    * Plotting features:
        * Plot the data of a specific ferrite material, e.g. the amplitude of the permeability over the magnetic flux density

    * Filter features:
        * Get all available data for specific filter keys (e.g. temperature, frequency, material etc.)
        * Filter for some specific value intervals (e.g. 10mT < B-flux < 30mT)
..


Installation
---------------

::

    pip install materialdatabase

..
    Basic usage and minimal example
    ------------------------------------

    Material properties:
    ::

        material_db = mdb.MaterialDatabase()
        materials = material_db.material_list_in_database()
        initial_u_r_abs = material_db.get_material_property(material_name="N95", property="initial_permeability")
        core_material_resistivity = material_db.get_material_property(material_name="N95", property="resistivity")

    .. image:: /docs/source/figures/database_json.png
       :align: center

    Interpolated permeability and permittivity data of a Material:

    ::

        b_ref, mu_r_real, mu_r_imag = material_db.permeability_data_to_pro_file(temperature=25, frequency=150000, material_name = "N95", datatype = "complex_permeability",
                                              datasource = mdb.MaterialDataSource.ManufacturerDatasheet, parent_directory = "")

        epsilon_r, epsilon_phi_deg = material_db.get_permittivity(temperature= 25, frequency=150000, material_name = "N95", datasource = "measurements",
                                              datatype = mdb.MeasurementDataType.ComplexPermittivity, measurement_setup = "LEA_LK",interpolation_type = "linear")

    These function return complex permittivity and permeability for a certain operation point defined by temperature and frequency.


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
