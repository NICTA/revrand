#! /usr/bin/env python3
""" simple_regression.py
    This demo shows how to construct a simple regression by composing a kernel
    and optimising its hyperparameters.
"""
import numpy as np
import matplotlib.pyplot as pl
import revrand.legacygp as gp
import revrand.legacygp.kernels  # NOQA
import logging
logging.basicConfig(level=logging.INFO)


def main():

    nTrain = 20
    nQuery = 100
    nDims = 1
    noise_level = 0.05

    # Test dataset----------------------------------------------
    X = np.random.uniform(0, 1.0, size=(nTrain, nDims))
    X = X[np.argsort(X[:, 0])]  # n*d
    underlyingFunc = (lambda x: np.sin(2*np.pi*x) + 5 +
                      np.random.normal(loc=0.0, scale=0.05,
                                       size=(x.shape[0], 1)))
    y = underlyingFunc(X) + noise_level * np.random.randn(nTrain, 1)
    y = y.ravel()
    Xs = np.linspace(0., 1., nQuery)[:, np.newaxis]
    data_mean = np.mean(y)
    ys = y - data_mean
    # ----------------------------------------------------------

    # Define a pathological GP kernel:
    def kerneldef(h, k):
        a = h(0.1, 5, 0.1)
        b = h(0.1, 5, 0.1)
        logsigma = h(-6, 1)
        return (a*k(gp.kernels.gaussian, b) + .1*b*k(gp.kernels.matern3on2, a)
                + k(gp.kernels.lognoise, logsigma))

    # Learning signal and noise hyperparameters
    hyper_params = gp.learn(X, ys, kerneldef, verbose=False, ftol=1e-15,
                            maxiter=2000)

    # old_hyper_params = [1.48, .322, np.log(0.0486)]
    # print(gp.describe(kerneldef, old_hyper_params))
    print(gp.describe(kerneldef, hyper_params))

    regressor = gp.condition(X, ys, kerneldef, hyper_params)
    query = gp.query(regressor, Xs)
    post_mu = gp.mean(query) + data_mean
    # post_cov = gp.covariance(query)
    post_var = gp.variance(query, noise=True)

    # Plot
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(Xs, post_mu, 'k-')
    upper = post_mu + 2*np.sqrt(post_var)
    lower = post_mu - 2*np.sqrt(post_var)

    ax.fill_between(Xs.ravel(), upper, lower,
                    facecolor=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5))
    ax.plot(regressor.X[:, 0], regressor.y+data_mean, 'r.')
    pl.show()


if __name__ == "__main__":
    main()
