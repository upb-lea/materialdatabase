# Material database for power electronic usage
The main purpose of the material database is to provide various materials for FEM simulations or other calculations in which material data from data sheets or own measurements are required. 

Possible application scenarios:
 - FEMMT loads the permeability or the conductivity of the core from the database, depending on the material.
 - Graphical user interface (GUI) in FEMMT can compare properties of the material stored in material database.
 
 
## Installation
```
pip install materialdatabase
```

## Basic usage and minimal example
Material properties:
```
material_db = mdb.MaterialDatabase()
materials = material_db.material_list_in_database()
initial_u_r_abs = material_db.get_material_property(material_name="N95", property="initial_permeability")
core_material_resistivity = material_db.get_material_property(material_name="N95", property="resistivity")
```
![image 1](/images/database_json.png)

Interpolated permeability and permittivity data of a Material:
```
b_ref, mu_r_real, mu_r_imag = material_db.permeability_data_to_pro_file(temperature=25, frequency=150000, material_name = "N95", datatype = "complex_permeability",
                                      datasource = mdb.MaterialDataSource.ManufacturerDatasheet, parent_directory = "")

epsilon_r, epsilon_phi_deg = material_db.get_permittivity(temperature= 25, frequency=150000, material_name = "N95", datasource = "measurements",
                                      datatype = mdb.MeasurementDataType.ComplexPermittivity, measurement_setup = "LEA_LK",interpolation_type = "linear")
```
These function return complex permittivity and permeability for a certain operation point defined by temperature and frequency.
## GUI (FEMMT)
The materials in database can be compared with help GUI in FEM magnetics toolbox. In database tab of GUI, the loss graphs and B-H curves from the datasheets of up to 5 materials can be compared.
 
FEMMT can be installed using the python pip package manager.

```
    pip install femmt
```
For working with the latest version, refer to the [documentation](https://upb-lea.github.io/FEM_Magnetics_Toolbox/main/intro.html).

![image 1](/images/gui_database.png)
![image 2](/images/gui_database_loss.png)