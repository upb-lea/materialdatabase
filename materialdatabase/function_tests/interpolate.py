import numpy as np
import scipy
from matplotlib import pyplot as plt

T = 45
f = 140000
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




result = interpolate_b_dependent_quantity_in_temperature_and_frequency(T, f, T_low, T_high, f_low, f_high,
                                                                       b_T_low_f_low, mu_T_low_f_low,
                                                                       b_T_high_f_low, mu_T_high_f_low,
                                                                       b_T_low_f_high, mu_T_low_f_high,
                                                                       b_T_high_f_high, mu_T_high_f_high)

print(result)
