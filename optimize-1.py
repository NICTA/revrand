import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pyalacarte.optimize import candidate_start_points_grid

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(*candidate_start_points_grid([(-1, 1.5), (-.5, 1.5), (0, 5)], [6, 8, 10]))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()