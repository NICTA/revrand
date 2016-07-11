""" Special Numerical Operations. """

from __future__ import division

import numpy as np


#
# Module constants
#

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
SMALL = 1e-100
LOGTINY = np.log(TINY)


#
# Special functions
#

def logsumexp(X, axis=0):
    """
    Log-sum-exp trick for matrix X for summation along a specified axis.

    This performs the following operation in a stable fashion,

    .. math::

        \log \sum^K_{k=1} \exp\{x_k\}

    Parameters
    ----------
        X: ndarray
            2D array of shape (N, D) to apply the log-sum-exp trick.
        axis: int, optional
            Axis to apply the summation along (works the same as axis in
            numpy.sum).

    Returns
    -------
        lseX: ndarray
            results of applying the log-sum-exp trick, this will be shape (D,)
            if :code:`axis=0` or shape (N,) if :code:`axis=1`.
    """

    mx = X.max(axis=axis)
    if (X.ndim > 1):
        mx = np.atleast_2d(mx).T if axis == 1 else np.atleast_2d(mx)

    return np.log(np.exp(X - mx).sum(axis=axis)) + np.ravel(mx)


def softmax(X, axis=0):
    """
    Pass X through a softmax function in a numerically stable way using the
    log-sum-exp trick.

    This transformation is:

    .. math::

        \\frac{\exp\{X_k\}}{\sum^K_{j=1} \exp\{X_j\}}

    and is appliedx to each row/column, `k`, of X.

    Parameters
    ----------
        X: ndarray
            2D array of shape (N, D) to apply the log-sum-exp trick.
        axis: int, optional
            Axis to apply the summation along (works the same as axis in
            numpy.sum).

    Returns
    -------
        smX: ndarray
            results of applying the log-sum-exp trick, this will be shape
            (N, D), and each row will sum to 1 if :code:`axis=1` or each column
            will sum to 1 if :code:`axis=0`.
    """

    if axis == 1:
        return np.exp(X - logsumexp(X, axis=1)[:, np.newaxis])
    elif axis == 0:
        return np.exp(X - logsumexp(X, axis=0))
    else:
        raise ValueError("This only works on 2D arrays for now.")


def softplus(X):
    """ Pass X through a soft-plus function, , in a numerically
        stable way (using the log-sum-exp trick).

        The softplus transformation is:

        .. math::
            \log(1 + \exp\{X\})

        Parameters
        ----------
            X: ndarray
                shape (N,) array or shape (N, D) array of data.

        Returns
        -------
            spX: ndarray
                array of same shape of X with the result of softmax(X).
    """

    if np.isscalar(X):
        return logsumexp(np.vstack((np.zeros(1), [X])).T, axis=1)[0]

    N = X.shape[0]

    if X.ndim == 1:
        return logsumexp(np.vstack((np.zeros(N), X)).T, axis=1)
    elif X.ndim == 2:
        sftX = np.empty(X.shape, dtype=float)
        for d in range(X.shape[1]):
            sftX[:, d] = logsumexp(np.vstack((np.zeros(N), X[:, d])).T, axis=1)
        return sftX
    else:
        raise ValueError("This only works on up to 2D arrays.")


#
# Numerically "safe" functions
#

def safelog(x, min_x=TINY):

    cx = x.copy()
    cx[cx < min_x] = min_x
    return np.log(cx)


def safesoftplus(x, min_x=SMALL):

    g = softplus(x)
    g[g < min_x] = min_x
    return g
