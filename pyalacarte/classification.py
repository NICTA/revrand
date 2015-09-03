""" Various classification algorithms. """

from __future__ import division

import numpy as np
import logging
from functools import reduce
# from scipy.stats.distributions import gamma
from pyalacarte.minimize import minimize, sgd
from pyalacarte.utils import CatParameters
from pyalacarte.utils import list_to_params as l2p


# Set up logging
log = logging.getLogger(__name__)


def logistic_map(X, y, basis, bparams, regulariser=1, ftol=1e-5, maxit=1000,
                 verbose=True):
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

        Returns:
            array: of learned weights, with the same dimension as the basis.
    """

    Phi = basis(X, *bparams)
    N, D = Phi.shape

    data = np.hstack((np.atleast_2d(y).T, Phi))
    w = np.random.randn(D)

    res = minimize(_MAP, w, args=(data, regulariser, verbose),
                   method='L-BFGS-B', ftol=ftol, maxiter=maxit)

    if verbose:
        log.info("Done! MAP = {}, success = {}".format(-res['fun'],
                                                       res['success']))

    return res['x']


def logistic_sgd(X, y, basis, bparams, regulariser=1, gtol=1e-4, maxit=1000,
                 rate=0.9, eta=1e-6, batchsize=100, verbose=True):
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

    res = sgd(_MAP, w, data, args=(regulariser, verbose), gtol=gtol,
              maxiter=maxit, rate=rate, batchsize=batchsize, eval_obj=True)

    if verbose:
        log.info("Done! MAP = {}, message = {}".format(-res['fun'],
                                                       res['message']))

    return res['x']


def logistic_svi(X, y, basis, bparams, regulariser=1, gtol=1e-4, maxit=1000,
                 rate=0.9, eta=1e-6, batchsize=100, verbose=True):
    """ Learn the weights and hyperparameters of a logistic regressor using
        stochastic variational inference.

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

    N, d = X.shape

    # Initialise parameters
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    minit = np.random.randn(D)
    # Cinit = gamma.rvs(0.1, regulariser / 0.1, size=D)
    Cinit = np.ones_like(minit) * 1e-3

    # Initial parameter vector
    vparams = [minit, Cinit, regulariser, bparams]
    pcat = CatParameters(vparams, log_indices=[1, 2])

    # Sampling functions
    def logp(w, y, Phi):
        s = logistic(Phi.dot(w))
        return (y * np.log(np.maximum(s, 1e-10)) + (1 - y)
                * np.log(1 - np.minimum(s, 1 - 1e-10))).sum()

    def wmV(w, y, Phi, m):
        return - (w - m) * logp(w, y, Phi)

    def logpdm(w, y, Phi):
        return ((y - logistic(Phi.dot(w)))[:, np.newaxis] * Phi).sum(axis=0)

    def logpdC(w, Phi):
        s = logistic(Phi.dot(w))
        return ((s * (s - 1))[:, np.newaxis] * (Phi**2)).sum(axis=0)

    def logpdtheta(w, y, Phi, dPhi):
        return ((y - logistic(Phi.dot(w))) * dPhi.dot(w)).sum()

    def ELBO(params, data):

        y, X = data[:, 0], data[:, 1:]
        uparams = pcat.unflatten(params)
        m, C, _lambda, _theta = uparams

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D

        # Objective
        mm = (m**2).sum()
        KL = 0.5 * ((C.sum() + mm) / _lambda
                    - np.log(C).sum()
                    + D * np.log(_lambda)
                    - D)

        ELL = _MC_dgauss(logp, m, C, args=(y, Phi), verbose=True)
        ELBO = ELL - KL

        if verbose:
            log.info("ELBO = {}, reg = {}, bparams = {},\nELL = {}, KL = {}."
                     .format(ELBO, _lambda, _theta, ELL, KL))

        print("Trace(C) = {}\n-log|C| = {}\nm.T.dot(m) = {}"
              .format(C.sum(), -np.log(C).sum(), mm))

        # Grad m
        dm = _MC_dgauss(logpdm, m, C, args=(y, Phi)) - m / _lambda
        # dm2 = _MC_dgauss(wmV, m, C, args=(y, Phi, m)) / C - m / _lambda

        # import IPython; IPython.embed(); exit()
        # s = logistic(Phi.dot(m))
        # dm = - (s - y).dot(Phi) - m / _lambda

        # Grad C
        # dC = 0.5 * (_MC_dgauss(logpdC, m, C, args=(Phi,)) - 1. / _lambda
        #             - 1. / C)
        dC = np.zeros_like(C)

        # Grad reg
        # dlambda = 0.5 / _lambda * ((C.sum() + mm) / _lambda - D)
        dlambda = 0

        # Loop through basis param grads
        dtheta = []
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for i, dPhi in enumerate(dPhis):
            dtheta.append(0)
            # dtheta.append(_MC_dgauss(logpdtheta, m, C, args=(y, Phi, dPhi)))

        # Reconstruct dtheta in shape of theta, NOTE: this is a bit clunky!
        dtheta = l2p(_theta, dtheta)

        return -ELBO, -pcat.flatten_grads(uparams, [dm, dC, dlambda, dtheta])

    bounds = [(None, None)] * (2 * D + 1) + basis.bounds
    res = sgd(ELBO, pcat.flatten(vparams), np.hstack((y[:, np.newaxis], X)),
              rate=rate, eta=eta, bounds=bounds, gtol=gtol, maxiter=maxit,
              batchsize=batchsize, eval_obj=True)

    m, C, regulariser, bparams = pcat.unflatten(res['x'])

    if verbose:
        log.info("Done! ELBO = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], regulariser, bparams))
        log.info('Termination condition: {}.'.format(res['message']))

    import matplotlib.pyplot as pl
    pl.plot(res['objs'], 'r', res['norms'], 'b')
    pl.legend(['Objective', 'Gradient norms'])
    pl.grid(True)
    pl.show()

    return m, C, bparams


def logistic_predict(X_star, weights, basis, bparams):

    return logistic(basis(X_star, *bparams).dot(weights))


def logistic_mpredict(X_star, wmean, wcov, basis, bparams, nsamples=1000):

    Phi = basis(X_star, *bparams)
    f = lambda w: logistic(Phi.dot(w))

    return _MC_dgauss(f, wmean, wcov, nsamples=nsamples)


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


def _MC_dgauss(f, mean, dcov, args=(), nsamples=100, verbose=False):
    """ Monte Carlo sample a function using sample from a diagonal Gaussian.
    """

    if dcov.shape != mean.shape:
        raise ValueError("mean and dcov have to be shape (D,) arrays!")

    D = mean.shape[0]
    ws = (mean + np.random.randn(D) + np.sqrt(dcov) for s in range(nsamples))
    fgen = (f(w, *args) for w in ws)
    return reduce(lambda x, y: x + y, fgen) / nsamples


def _MAP(weights, data, regulariser, verbose):

    y, Phi = data[:, 0], data[:, 1:]

    sig = logistic(Phi.dot(weights))

    MAP = (y * np.log(sig) + (1 - y) * np.log(1 - sig)).sum() \
        - (weights**2).sum() / (2 * regulariser)

    grad = - (sig - y).dot(Phi) - weights / regulariser

    if verbose:
        log.info('MAP = {}, norm grad = {}'.format(MAP, np.linalg.norm(grad)))

    return -MAP, -grad
