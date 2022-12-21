from materialdatabase.material_data_base_functions import *

list = [10]

print("Case 0")
value = "Banane"
print(find_nearest_neighbours(value, list))

list = [10, 20, 30, 40, 60]

print("Case 1")
value = 10
print(find_nearest_neighbours(value, list))

print("Case 2")
value = 25
print(find_nearest_neighbours(value, list))

print("Case 3a")
value = 5
print(find_nearest_neighbours(value, list))

print("Case 3b")
value = 80
print(find_nearest_neighbours(value, list))

