import matplotlib.pyplot as plt
import numpy as np

from revrand.utils.datasets import make_polynomial

deg = 3

X, y, coefs = make_polynomial(degree=deg, n_samples=200, noise=.5,
                       return_coefs=True, random_state=1)

poly = np.vectorize(lambda x: np.sum(coefs * x ** np.arange(deg+1)))
x = np.arange(-3, 3, 0.2)
y_true = poly(x)

fig, ax = plt.subplots()

label = ', '.join(r'$w_{0}={1}$'.format(*coef) for coef in enumerate(coefs.round(2)))

ax.plot(x, y_true, 'r-', label=label)
ax.scatter(X, y, alpha=0.4, label=r'Noise $\beta=0.5$')

ax.legend(loc='upper left')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()