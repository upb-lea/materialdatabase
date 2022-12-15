# Material database for power electronic usage
The main purpose of the material database is to provide various materials for FEM simulations or other calculations in which material data from data sheets or own measurements are required. 

Possible application scenarios:
 - FEMMT loads the permeability or the conductivity of the core from the database, depending on the material.
 - ...
 
 
## Installation
```
pip install materialdatabase
```

## Basic usage and minimal example
```
    database = mdb.MaterialDatabase()
    initial_u_r = material_db.get_material_property(material_name="N95", property="initial_permeability")
```

