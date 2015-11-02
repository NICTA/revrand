""" Various nonlinear transformation functions. """

import numpy as np


def logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    if (X.ndim > 1):
        mx = np.atleast_2d(mx).T

    return np.log(np.exp(X - mx).sum(axis=axis)) + np.ravel(mx)


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
            lgX[:, d] = np.exp(-logsumexp(np.vstack((np.zeros(N),
                                                     -X[:, d])).T, axis=1))
        return lgX
    else:
        raise ValueError("This only works on up to 2D arrays.")


def softmax(X, axis=0):
    """ Pass X through a softmax function, exp(X) / sum(exp(X), axis=axis), in
        a numerically stable way using the log-sum-exp trick.
    """

    if axis == 1:
        return np.exp(X - logsumexp(X, axis=1)[:, np.newaxis])
    elif axis == 0:
        return np.exp(X - logsumexp(X, axis=0))
    else:
        raise ValueError("This only works on 2D arrays for now.")
