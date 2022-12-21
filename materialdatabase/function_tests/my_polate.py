from materialdatabase.material_data_base_functions import *

a = 10
b = 20
f_a = 100
f_b = 200

# interpolate
x = 15
print(my_polate_linear(a, b, f_a, f_b, x))

# low extrapolate
x = 5
print(my_polate_linear(a, b, f_a, f_b, x))

# negative low extrapolate
x = -5
print(my_polate_linear(a, b, f_a, f_b, x))

# high extrapolate
x = 25
print(my_polate_linear(a, b, f_a, f_b, x))

# high exact: x=b
x = b
print(my_polate_linear(a, b, f_a, f_b, x))

# low exact: x=a
x = a
print(my_polate_linear(a, b, f_a, f_b, x))



a = 10
b = 10
f_a = 100
f_b = 100

# interpolate
x = 10
print(my_polate_linear(a, b, f_a, f_b, x))

