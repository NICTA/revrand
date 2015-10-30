import matplotlib.pyplot as plt
import numpy as np

from revrand.utils.datasets import make_regression

f = lambda x: 0.5 * x + np.sin(2 * x)

x = np.arange(-3, 3, 0.2)
y_true = f(x)

X, y = make_regression(f, n_samples=200, noise=0.15, random_state=1)

fig, ax = plt.subplots()

ax.plot(x, y_true, 'r-', label=r'True')
ax.scatter(X, y, alpha=0.4, label=r'Noise $\beta=0.15$')

ax.legend(loc='upper left')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()