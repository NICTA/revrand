""" Various classification algorithms. """

from __future__ import division

import numpy as np
import logging
from pyalacarte.minimize import minimize, sgd


# Set up logging
log = logging.getLogger(__name__)


def logistic_map(X, y, basis, bparams, regulariser=1e-3, ftol=1e-5, maxit=1000,
                 verbose=True, regulariser_bounds=(1e-7, None)):
    """ Learn the weights of a logistic regressor using MAP inference.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array of boolean targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of parameters of the basis object
            regulariser, (float): weight regulariser (variance) initial guess
            ftol, (float): optimiser function tolerance convergence criterion
            maxit, (int): maximum number of iterations for SGD
            verbose, (float): log learning status
            regulariser_bounds, (tuple): of (lower bound, upper bound) on the
                regulariser parameter, None for unbounded (though it cannot be
                <= 0)

        Returns:
            array: of learned weights, with the same dimension as the basis.
    """

    Phi = basis(X, *bparams)
    N, D = Phi.shape

    data = np.hstack((np.atleast_2d(y).T, Phi))
    w = np.random.randn(D)

    res = minimize(MAP, w, args=(data, regulariser, verbose),
                   method='L-BFGS-B', ftol=ftol, maxiter=maxit)

    if verbose:
        log.info("Done! MAP = {}, success = {}".format(-res['fun'],
                                                       res['success']))

    return res['x']


def logistic_sgd(X, y, basis, bparams, regulariser=1e-3, gtol=1e-5, maxit=1000,
                 rate=0.5, batchsize=100, verbose=True,
                 regulariser_bounds=(1e-7, None)):
    """ Learn the weights of a logistic regressor using MAP inference and SGD.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array of boolean targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of parameters of the basis object
            regulariser, (float): weight regulariser (variance) initial guess
            gtol, (float): SGD tolerance convergence criterion
            maxit, (int): maximum number of iterations for SGD
            rate, (float): SGD learing rate.
            batchsize, (int): number of observations to use per SGD batch.
            verbose, (float): log learning status
            regulariser_bounds, (tuple): of (lower bound, upper bound) on the
                regulariser parameter, None for unbounded (though it cannot be
                <= 0)

        Returns:
            array: of learned weights, with the same dimension as the basis.
    """

    Phi = basis(X, *bparams)
    N, D = Phi.shape

    data = np.hstack((np.atleast_2d(y).T, Phi))
    w = np.random.randn(D)

    res = sgd(MAP, w, data, args=(regulariser, verbose), gtol=gtol,
              maxiter=maxit, rate=rate, batchsize=batchsize, eval_obj=True)

    if verbose:
        log.info("Done! MAP = {}, message = {}".format(-res['fun'],
                                                       res['message']))

    return res['x']


def logistic_predict(X_star, weights, basis, bparams):

    return logistic(basis(X_star, *bparams).dot(weights))


def MAP(weights, data, regulariser, verbose):

    y, Phi = data[:, 0], data[:, 1:]

    sig = logistic(Phi.dot(weights))

    MAP = (y * np.log(sig) + (1 - y) * np.log(1 - sig)).sum() \
        - (weights**2).sum() / (2 * regulariser)

    grad = - (sig - y).dot(Phi) - weights / regulariser

    if verbose:
        log.info('MAP = {}, norm grad = {}'.format(MAP, np.linalg.norm(grad)))

    return -MAP, -grad


def logistic(X):
    """ Pass X through a logistic sigmoid, 1 / (1 + exp(-X)), in a numerically
        stable way (using the log-sum-exp trick).

        Arguments:
            X: shape (N,) array or shape (N, D) array of data.

        Returns:
            array of same shape of X with the result of logistic(X).
    """

    N = X.shape[0]

    if X.ndim == 1:
        return np.exp(-logsumexp(np.vstack((np.zeros(N), -X)).T, axis=1))
    elif X.ndim == 2:
        lgX = np.empty(X.shape, dtype=float)
        for d in range(X.shape[1]):
            lgX[:, d] = np.exp(-logsumexp(np.vstack((np.zeros(N), -X[:, d])).T,
                               axis=1))
        return lgX
    else:
        raise ValueError("This only works on up to 2D arrays.")


def logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    return np.log(np.exp(X - mx[:, np.newaxis]).sum(axis=axis)) + mx

