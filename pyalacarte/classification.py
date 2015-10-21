""" Various classification algorithms. """

from __future__ import division

import numpy as np
import logging

from .optimize import minimize, sgd


# Set up logging
log = logging.getLogger(__name__)


def logistic_map(X, y, basis, bparams, regulariser=1., balance=True, ftol=1e-6,
                 maxit=1000, verbose=True):
    """
    Learn the weights of a multiclass logistic regressor using MAP inference.

    Parameters
    ----------
        X: ndarray
            (N, D) array input dataset (N samples, D dimensions).
        y: ndarray
            (N,) array of integer targets (N samples).
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        regulariser: float, optional
            weight regulariser (variance) initial values.
        balance: bool, optional
            Automatically accout for unbalanced classes, i.e. adjust the
            contibution of the classes to the objective function accorging to
            their size in the dataset.
        ftol: float, optional
            optimiser function tolerance convergence criterion.
        maxit: int, optional
            maximum number of iterations for the optimiser.
        verbose: float, optional
            log learning status.

    Returns
    -------
        weights: ndarray
            learned weights, with the same dimension as the basis.
        labels: sequence
            the order of the class labels in this sequence is the same as they
            will appear in the predictive output (which is a matrix of
            probabilities).
    """

    # Parse input labels
    labels, yu, cweights, K = _parse_labels(y, balance)

    Phi = basis(X, *bparams)
    N, D = Phi.shape

    data = np.hstack((np.atleast_2d(yu).T, Phi))
    W = np.random.randn(D, K).flatten()

    res = minimize(_MAP, W, args=(data, regulariser, cweights, verbose),
                   method='L-BFGS-B', ftol=ftol, maxiter=maxit, jac=True)

    if verbose:
        log.info("Done! MAP = {}, success = {}".format(-res.fun, res.success))

    return res.x.reshape((D, K)), labels


def logistic_sgd(X, y, basis, bparams, regulariser=1, balance=True, gtol=1e-4,
                 passes=100, rate=0.9, eta=1e-6, batchsize=100, verbose=True):
    """
    Learn the weights of a logistic regressor using MAP inference and SGD.

    Parameters
    ----------
        X: ndarray
            (N, D) array input dataset (N samples, D dimensions).
        y: ndarray
            (N,) array of boolean targets (N samples).
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        regulariser: float, optional
            weight regulariser (variance) initial values.
        balance: bool, optional
            Automatically accout for unbalanced classes, i.e. adjust the
            contibution of the classes to the objective function accorging to
            their size in the dataset.
        gtol: float, optional
            SGD tolerance convergence criterion.
        passes: int, optional
            Number of complete passes through the data before optimization
            terminates (unless it converges first).
        rate: float, optional
            SGD learing rate.
        batchsize: int, optional
            number of observations to use per SGD batch.
        verbose: float, optional
            log learning status

    Returns
    -------
        weights: ndarray
            learned weights, with the same dimension as the basis.
        labels: sequence
            the order of the class labels in this sequence is the same as they
            will appear in the predictive output (which is a matrix of
            probabilities).
    """

    # Parse input labels
    labels, yu, cweights, K = _parse_labels(y, balance)

    Phi = basis(X, *bparams)
    N, D = Phi.shape

    data = np.hstack((np.atleast_2d(yu).T, Phi))
    W = np.random.randn(D, K).flatten()

    res = sgd(_MAP, W, data, args=(regulariser, cweights, verbose, N),
              gtol=gtol, passes=passes, rate=rate, batchsize=batchsize,
              eval_obj=True)

    if verbose:
        log.info("Done! MAP = {}, message = {}".format(-res.fun, res.message))

    return res.x.reshape((D, K)), labels


def logistic_predict(X_star, weights, basis, bparams):
    """
    Predict using multiclass logisitic regression (MAP).

    Parameters
    ----------
        X_star: ndarray
            (N_star, D) array query input dataset (N_star samples, D
             dimensions).
        weights: ndarray
            (D', K) array of regression weights, where D' is the dimension of
            the basis, and K is the number of classes.
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of hyperparameters of the basis object.

    Returns
    -------
        Prob_y: ndarray
            A (N_star, K) matrix of the probabilites of each query input
            belonging to a particular class. The column orders corresponds to
            the `label` order output by the learning function.
    """

    return _softmax(basis(X_star, *bparams).dot(weights), axis=1)


#
# Private module functions
#

def _logistic(X):
    """ Pass X through a logistic sigmoid, 1 / (1 + exp(-X)), in a numerically
        stable way (using the log-sum-exp trick).

        Arguments:
            X: shape (N,) array or shape (N, D) array of data.

        Returns:
            array of same shape of X with the result of logistic(X).
    """

    N = X.shape[0]

    if X.ndim == 1:
        return np.exp(-_logsumexp(np.vstack((np.zeros(N), -X)).T, axis=1))
    elif X.ndim == 2:
        lgX = np.empty(X.shape, dtype=float)
        for d in range(X.shape[1]):
            lgX[:, d] = np.exp(-_logsumexp(np.vstack((np.zeros(N),
                                                      -X[:, d])).T, axis=1))
        return lgX
    else:
        raise ValueError("This only works on up to 2D arrays.")


def _softmax(X, axis=0):
    """ Pass X through a softmax function, exp(X) / sum(exp(X), axis=axis), in
        a numerically stable way using the log-sum-exp trick.
    """

    if axis == 1:
        return np.exp(X - _logsumexp(X, axis=1)[:, np.newaxis])
    elif axis == 0:
        return np.exp(X - _logsumexp(X, axis=0))
    else:
        raise ValueError("This only works on 2D arrays for now.")


def _logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    return np.log(np.exp(X - mx[:, np.newaxis]).sum(axis=axis)) + mx


def _MAP(weights, data, regulariser, cweights, verbose, N=None):
    y, Phi = data[:, 0], data[:, 1:]
    scale = 1 if N is None else len(y) / N

    D, K = Phi.shape[1], int(y.max() + 1)
    weights = weights.reshape((D, K))
    if cweights is None:
        cweights = np.ones(K)

    sig = _softmax(Phi.dot(weights), axis=1)
    grad = np.zeros_like(weights)
    MAP = 0

    for k in range(K):
        yk = (y == k)
        MAP += cweights[k] * (np.log(sig[yk, k])).sum() \
            - scale * (weights[:, k]**2).sum() / (2 * regulariser)
        grad[:, k] = cweights[k] * (yk - sig[:, k]).dot(Phi) \
            - scale * weights[:, k] / regulariser

    if verbose:
        log.info('MAP = {}, norm grad = {}'.format(MAP, np.linalg.norm(grad)))

    return -MAP, -grad.flatten()


def _parse_labels(y, balance):

    # Parse input labels
    labels, yu = np.unique(y, return_inverse=True)
    counts = np.bincount(yu)
    cweights = counts.mean() / counts if balance else None
    K = len(labels)

    return labels, yu, cweights, K
