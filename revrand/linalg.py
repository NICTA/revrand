"""
Various linear algebra utilities.

Notes on the fast Hadamard transform (Alistair Reid, NICTA 2015):

The Hadamard transform is a recursive application of sums and differences
on a vector of length 2^n. The log(n) recursive application allow computation
in n log(n) operations instead of the naive n^2 per vector that would result
from computing and then multiplying a Hadamard matrix.

Note: the Walsh ordering is naturally computed by the recursive algorithm. The
sequence ordering requires a re-ordering afterwards incurring a similar
overhead to the original computation.

.. code:: python

    # Larger tests against calling Julia: [length 4*512 vectors:]
    M = np.argsort(np.sin(np.arange(64))).astype(float)
        Julia: 570us (sequence ordered)
        Al's single optimised: 560us (sequence ordered), 30us (natural)
        Al's basic vectorised: 1370us (sequence ordered)
"""

from __future__ import division

import numpy as np
from math import log
from scipy.linalg import cholesky, LinAlgError


def jitchol(a, jit=None, jit_max=1e-3, returns_jit=False, lower=False,
            overwrite_a=False, check_finite=True):
    """
    Do cholesky decomposition with a bit of diagonal jitter if needs be.

    Arguments:
        A: a [NxN] positive definite symmetric matrix to be decomposed as
            A = L.dot(L.T).
        lower: Return lower triangular factor, default False (upper).

    Returns:
        An upper or lower triangular matrix factor, L, also [NxN].
        Also wheter or not a the matrix is lower triangular form,
        (L, lower).

    Examples
    --------
    >>> a = np.array([[1, -2j],
    ...               [2j, 5]])
    >>> jitchol(a, lower=True)
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> np.all(a == np.array([[1, -2j],
    ...                       [2j, 5]]))
    True

    >>> b = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])
    >>> U, jit = jitchol(b, returns_jit=True)
    >>> U.round(2)
    array([[ 1.41, -0.71,  0.  ],
           [ 0.  ,  1.22, -0.82],
           [ 0.  ,  0.  ,  1.15]])
    >>> jit is None
    True

    Should remain unchanged

    >>> b
    array([[ 2, -1,  0],
           [-1,  2, -1],
           [ 0, -1,  2]])

    >>> c = np.array([[1, 2],
    ...               [2, 1]])
    >>> jitchol(c) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    LinAlgError: Exceeded maximum jitter limit, yet a is still not positive
    semidefinite!
    """

    try:
        chol = cholesky(a, lower=lower, overwrite_a=overwrite_a,
                        check_finite=check_finite)
        if returns_jit:
            return chol, jit
        else:
            return chol

    except LinAlgError:

        if jit is None:
            jit = 1e-16

        if jit > jit_max:
            raise LinAlgError('Exceeded maximum jitter limit, yet a is still'
                              ' not positive semidefinite!')

        diag = np.diag(a)
        diag_mean = diag.mean()
        diag_delta = jit * diag_mean

        if overwrite_a:
            diag_ind = np.diag_indices_from(a)
            a[diag_ind] += diag_delta
            return jitchol(a, jit=10 * jit, jit_max=jit_max,
                           returns_jit=returns_jit, lower=lower,
                           overwrite_a=overwrite_a, check_finite=check_finite)

        return jitchol(a + diag_delta * np.eye(*a.shape), jit=10 * jit,
                       jit_max=jit_max, returns_jit=returns_jit, lower=lower,
                       overwrite_a=overwrite_a, check_finite=check_finite)


def cho_log_det(c):
    """
    Compute the log of the determinant of `A`, given its (upper or lower)
    Cholesky factorization `c`.

    Examples
    --------
    >>> a = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])

    >>> u = cholesky(a)
    >>> np.isclose(cho_log_det(u), np.log(np.linalg.det(a)))
    True

    >>> l = cholesky(a, lower=True)
    >>> np.isclose(cho_log_det(l), np.log(np.linalg.det(a)))
    True
    """
    return 2 * np.sum(np.log(c.diagonal()))


def hadamard(Y, ordering=True):
    """ *Very fast* Hadamard transform for single vector at a time

    Parameters
    ----------
    Y: ndarray
        the n*2^k data to be 1d hadamard transformed
    ordering: bool, optional
        reorder from Walsh to sequence

    Returns
    -------
    H: ndarray
        hadamard transformed data.
    """
    # dot is a sum product over the last axis of a and the second-to-last of b.
    # Transpose - can specify axes, default transposes a[0] and a[1] hmm
    n_vectors, n_Y = Y.shape
    matching = (n_vectors, 2, int(n_Y / 2))
    H = np.array([[1, 1], [1, -1]]) / 2.  # Julia uses 2 and not sqrt(2)?
    steps = int(log(n_Y) / log(2))
    assert(2**steps == n_Y)  # required
    for _ in range(steps):
        Y = np.transpose(Y.reshape(matching), (0, 2, 1)).dot(H)
    Y = Y.reshape((n_vectors, n_Y))
    if ordering:
        Y = Y[:, _sequency(n_Y)]
    return Y


def hadamard_basic(Y, ordering=True):
    """
    Fast Hadamard transform using vectorised indices

    Parameters
    ----------
    Y: ndarray
        the n*2^k data to be 1d hadamard transformed
    ordering: bool, optional
        reorder from Walsh to sequence

    Returns
    -------
    H: ndarray
        hadamard transformed data.
    """

    Y = Y.T
    n = len(Y) / 2
    r = np.arange(len(Y))
    while n >= 1:
        ind = (r / n).astype(int) % 2 == 0
        a = (Y[ind] + Y[~ind]) / 2.
        b = (Y[ind] - Y[~ind]) / 2.
        Y[ind] = a
        Y[~ind] = b
        n /= 2
    if ordering:
        Y = Y[_sequency(len(Y))]
    return Y.T


def _sequency(length):
    # http://fourier.eng.hmc.edu/e161/lectures/wht/node3.html
    # Although this incorrectly explains grey codes...
    s = np.arange(length).astype(int)
    s = (s >> 1) ^ s  # Grey code ...
    # Reverse bits
    order = np.zeros(s.shape).astype(int)
    n = int(1)
    m = length // 2
    while n < length:
        order |= m * (n & s) // n
        n <<= 1
        m >>= 1
    return order
