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
initial_u_r = material_db.get_material_property(material_name="N95", property="initial_permeability")
resistivity = material_db.get_material_property(material_name="N95", property="resistivity")
Steinmetz_data = material_db.get_steinmetz_data(material_name="N95", type="Steinmetz", datasource="measurements")
```
![image 1](/images/database_json.png)

Interpolated permeability and permittivity data of a Material:
```
permeability_data = permeability_data_to_pro_file(self, T=25, f=150000, material_name = "N95", datatype = "complex_permeability",
                                      datasource = "manufacturer_datasheet", measurement_setup = str, parent_directory = "")

permittivity_data = get_permittivity(self, T= 25, f=150000, material_name = "N95", datasource = "measurements", 
                                      datatype = "complex_permittivity", measurement_setup = str,interpolation_type = "linear")
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