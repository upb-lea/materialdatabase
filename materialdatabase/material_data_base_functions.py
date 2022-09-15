# all static functions shall be inserted in this file
import numpy as np
import json
import os
from scipy.interpolate import interp1d


# ------Remove Duplicate from freq array------
def remove(arr, n):
    mp = {i: 0 for i in arr}
    for i in range(n):
        if mp[arr[i]] == 0:
            mp[arr[i]] = 1
            return mp


# -----find nearby frequency n Temp---------
def find_nearest(array, value):
    array = np.asarray(array)
    array.sort()
    idx = (np.abs(array - value)).argmin()
    if array[idx] > value:
        return array[idx - 1], array[idx]
    else:
        return array[idx], array[idx + 1]


# -------interpolation function-------------
def interpolate(x1, x2, y1, y2):
    curve_1 = interp1d(x1, y1)
    curve_2 = interp1d(x2, y2)



