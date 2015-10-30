import matplotlib.pyplot as plt
import numpy as np

from revrand.utils.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D

f = lambda x0, x1: (x1 - 5) ** 2 + .5 * x0 * x1 + .25 * (x0 + 4) ** 2

x1, x0 = np.mgrid[-5:5:0.2, -5:5:0.2]
y_true = f(x0, x1)

X, y = make_regression(f, n_samples=200, n_features=2, noise=0.15,
                       random_state=5)

fig = plt.figure()
ax = plt.axes(projection='3d', azim=50)

ax.plot_surface(x0, x1, y_true, rstride=3, cstride=3, cmap=plt.cm.jet,
                alpha=0.2)
ax.scatter(X[:, 0], X[:, 1], y, alpha=0.6)

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$f(x_0, x_1)$')

plt.show()