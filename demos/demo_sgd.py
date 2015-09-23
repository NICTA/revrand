#! /usr/bin/env python3
""" A Small demo to try out SGD on a Maximum likelihood regression problem. """


import numpy as np
import matplotlib.pyplot as pl
from pyalacarte.minimize import minimize, sgd
from pyalacarte.bases import RadialBasis


# Objective function
def f(w, Data, sigma=1.0):

    y, Phi = Data[:, 0], Data[:, 1:]

    logp = -1.0 / (2 * sigma * sigma) * ((y - Phi.dot(w))**2).sum()
    d = 1.0 / (sigma * sigma) * Phi.T.dot((y - Phi.dot(w)))

    return -logp, -d


def sgd_demo():
    # Settings

    batchsize = 100
    var = 0.05
    nPoints = 1000
    nQueries = 500
    passes = 200
    min_grad_norm = 0.01
    rate = 0.9
    eta = 1e-5

    # Create dataset
    X = np.linspace(0.0, 1.0, nPoints)[:, np.newaxis]
    Y = np.sin(2 * np.pi * X.flatten()) + np.random.randn(nPoints) * var
    centres = np.linspace(0.0, 1.0, 20)[:, np.newaxis]
    Phi = RadialBasis(centres)(X, 0.1)
    train_dat = np.hstack((Y[:, np.newaxis], Phi))

    Xs = np.linspace(0.0, 1.0, nQueries)[:, np.newaxis]
    Yt = np.sin(2 * np.pi * Xs.flatten())
    Phi_s = RadialBasis(centres)(Xs, 0.1)
    w = np.linalg.solve(Phi.T.dot(Phi), Phi.T.dot(Y))
    Ys = Phi_s.dot(w)

    # L-BFGS approach to test objective
    w0 = np.random.randn(Phi.shape[1])
    results = minimize(f, w0, args=(train_dat,), jac=True, method='L-BFGS-B')
    w_grad = results['x']
    Ys_grad = Phi_s.dot(w_grad)

    # SGD for learning w
    w0 = np.random.randn(Phi.shape[1])
    results = sgd(f, w0, train_dat, passes=passes, batchsize=batchsize,
                  eval_obj=True, gtol=min_grad_norm, rate=rate, eta=eta)
    w_sgd, gnorms, costs = results['x'], results['norms'], results['objs']

    Ys_sgd = Phi_s.dot(w_sgd)

    # Visualise results
    fig = pl.figure()
    ax = fig.add_subplot(121)
    # truth
    pl.plot(X, Y, 'r.', Xs, Yt, 'k-')
    # exact weights
    pl.plot(Xs, Ys, 'c-')
    pl.plot(Xs, Ys_grad, 'b-')
    pl.plot(Xs, Ys_sgd, 'g-')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(['Training', 'Truth', 'Analytic', 'LBFGS', 'SGD'])

    ax = fig.add_subplot(122)
    pl.xlabel('iteration')
    ax.plot(range(len(costs)), costs, 'b')
    ax.set_ylabel('cost', color='b')
    for t in ax.get_yticklabels():
        t.set_color('b')
    ax2 = ax.twinx()
    ax2.plot(range(len(gnorms)), gnorms, 'r')
    ax2.set_ylabel('gradient norms', color='r')
    for t in ax2.get_yticklabels():
        t.set_color('r')

    pl.show()


if __name__ == "__main__":
    sgd_demo()
