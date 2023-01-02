import numpy as np
import scipy
from matplotlib import pyplot as plt
x_ = np.array([1, 2, 3, 4, 2, 2.5, 3])
y_ = np.array([2, 2, 4, 4, 3.25, 3.25, 1])
z_ = np.array([7, 9, 12, 13, 11, 11.4, 4])


x, y, z = x_, y_, z_
print(f"{x.shape, y.shape, z.shape = }")

f = scipy.interpolate.interp2d(x_, y_, z_)
print(f(3.4, 3.4))


density = 100
x_dense = np.linspace(min(x_), max(x_), density)
y_dense = np.linspace(min(y_), max(y_), density)
z_dense = f(x_dense, y_dense)
print(f"{x_dense.shape, y_dense.shape, z_dense.shape = }")
print(f"{z_dense.max(), z_dense.min() = }")
# print(f"{z_dense = }")


fig, ax = plt.subplots(nrows=1, ncols=1)

# Plot interpolated data at first, so it will be in the background of the plot
interpolated_data = ax.pcolormesh(x_dense, y_dense, z_dense)
fig.colorbar(interpolated_data)

# The input points obviously dissapper for good interpolation
input_data = ax.scatter(x, y, c=z, s=1000, marker="o", vmin=z_dense.min(), vmax=z_dense.max(), linewidths=.5, edgecolors="red")
# fig.colorbar(input_data)  # can be plotted for prove, that the exact same colout plot is used


plt.show()