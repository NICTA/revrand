#! /usr/bin/env python3
""" regression_2D.py
    This code is a demo of how to compose and use a basic GP kernel with
    multi-dimensional inputs, and provides a suggested visualisation for
    showing both the predictive mean and the predictive uncertainty.
"""

import numpy as np
import matplotlib.pyplot as pl
import revrand.legacygp as gp
import revrand.legacygp.kernels  # NOQA
import logging
logging.basicConfig(level=logging.INFO)


def main():

    nTrain = 200
    nQuery = [150, 100]
    nDims = 2

    # Make test dataset:
    X = np.random.uniform(0, 1.0, size=(nTrain, nDims))
    X = (X + 0.2 * (X > 0.5))/1.2
    X = X[np.argsort(X[:, 0])]
    noise = np.random.normal(loc=0.0, scale=0.1, size=(nTrain,))
    Y = (np.cos(3*np.pi*X[:, 0]) + np.cos(3*np.pi*X[:, 1])) + noise
    Y = Y[:, np.newaxis]

    data_mean = np.mean(Y, axis=0)
    ys = Y-data_mean

    # make a gridded query
    Xsx = np.linspace(0., 1., nQuery[0])
    Xsy = np.linspace(0., 1., nQuery[1])
    xv, yv = np.meshgrid(Xsx, Xsy)
    Xs = np.vstack((xv.ravel(), yv.ravel())).T

    # Compose isotropic kernel:
    def kernel_defn(h, k):
        a = h(0.1, 5, 0.1)
        b = h(0.1, 5, 0.1)
        logsigma = h(-6, 1)
        return a*k(gp.kernels.gaussian, b) + k(gp.kernels.lognoise, logsigma)

    hyper_params = gp.learn(X, ys, kernel_defn, verbose=True, ftol=1e-5,
                            maxiter=2000)

    print(gp.describe(kernel_defn, hyper_params))

    regressor = gp.condition(X, ys, kernel_defn, hyper_params)

    query = gp.query(regressor, Xs)
    post_mu = gp.mean(query) + data_mean
    post_var = gp.variance(query, noise=True)

    # Shift outputs back:
    ax = pl.subplot(131)
    pl.scatter(X[:, 0], X[:, 1], s=20, c=Y, linewidths=0)
    pl.axis('equal')
    pl.title('Training')
    pl.subplot(132, sharex=ax, sharey=ax)
    pl.scatter(Xs[:, 0], Xs[:, 1], s=20, c=post_mu, linewidths=0)
    pl.axis('equal')
    pl.title('Prediction')
    pl.subplot(133, sharex=ax, sharey=ax)
    pl.scatter(Xs[:, 0], Xs[:, 1], s=20, c=np.sqrt(post_var), linewidths=0)
    pl.axis('equal')
    pl.title('Stdev')
    pl.tight_layout()
    pl.show()


if __name__ == "__main__":
    main()
