# all static functions shall be inserted in this file

# Python integrated libraries

# 3rd party libraries
import numpy as np

# local libraries

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
        if idx == len(array) - 1:
            return array[idx - 1], array[idx]
        else:
            return array[idx], array[idx + 1]


def set_silent_status(is_silent: bool):
    """
    Silent mode global variable.

    :param is_silent: True for silent mode, False for mode with print outputs
    :type is_silent: bool
    """
    global silent
    silent = is_silent


def mdb_print(text: str, end='\n'):
    """
    Print function what checks the silent-mode-flag.
    Print only in case of no-silent-mode.

    :param text: Text to print
    :type text: str
    :param end: command for end of line, e.g. '\n' or '\t'
    :type end: str

    """
    if not silent:
        print(text, end)


def rect(r, theta_deg):
    """
    converts polar coordinates [degree] kartesian coordinates
    theta in degrees
    :param r: radius or amplitude
    :param theta_deg: angle in degree

    :returns: tuple; (float, float); (x,y)
    """
    x = r * np.cos(np.radians(theta_deg))
    y = r * np.sin(np.radians(theta_deg))
    return x, y
