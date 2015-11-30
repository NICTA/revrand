""" Various classification algorithms. """

from __future__ import division

import numpy as np
import logging

from .optimize import minimize, sgd
from .transforms import softmax


# Set up logging
log = logging.getLogger(__name__)


def learn_map(X, y, basis, bparams, regulariser=1., balance=True, ftol=1e-6,
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


def learn_sgd(X, y, basis, bparams, regulariser=1, balance=True, gtol=1e-4,
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


def predict(X_star, weights, basis, bparams):
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

    return softmax(basis(X_star, *bparams).dot(weights), axis=1)


#
# Private module functions
#

def _MAP(weights, data, regulariser, cweights, verbose, N=None):
    y, Phi = data[:, 0], data[:, 1:]
    scale = 1 if N is None else len(y) / N

    D, K = Phi.shape[1], int(y.max() + 1)
    weights = weights.reshape((D, K))
    if cweights is None:
        cweights = np.ones(K)

    sig = softmax(Phi.dot(weights), axis=1)
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
