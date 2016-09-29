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
from scipy.linalg import cholesky, cho_solve, svd, LinAlgError


# Numerical constants / thresholds
CHOLTHRESH = 1e-5


def cho_log_det(L):
    """
    Compute the log of the determinant of :math:`A`, given its (upper or lower)
    Cholesky factorization :math:`LL^T`.

    Parameters
    ----------
    L: ndarray
        an upper or lower Cholesky factor

    Examples
    --------
    >>> A = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])

    >>> Lt = cholesky(A)
    >>> np.isclose(cho_log_det(Lt), np.log(np.linalg.det(A)))
    True

    >>> L = cholesky(A, lower=True)
    >>> np.isclose(cho_log_det(L), np.log(np.linalg.det(A)))
    True
    """
    return 2 * np.sum(np.log(L.diagonal()))


def svd_log_det(s):
    """
    Compute the log of the determinant of :math:`A`, given its singular values
    from an SVD factorisation (i.e. :code:`s` from :code:`U, s, Ut = svd(A)`).

    Parameters
    ----------
    s: ndarray
        the singular values from an SVD decomposition

    Examples
    --------
    >>> A = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])

    >>> _, s, _ = np.linalg.svd(A)
    >>> np.isclose(svd_log_det(s), np.log(np.linalg.det(A)))
    True
    """
    return np.sum(np.log(s))


def solve_posdef(A, b):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix.

    This first tries cholesky, and if numerically unstable with solve using a
    truncated SVD (see solve_posdef_svd).

    The log-determinant of :math:`A` is also returned since it requires minimal
    overhead.

    Parameters
    ----------
    A: ndarray
        A positive semi-definite matrix.
    b: ndarray
        An array or matrix

    Returns
    -------
    X: ndarray
        The result of :math:`X = A^-1 b`
    logdet: float
        The log-determinant of :math:`A`
    """
    # Try cholesky for speed
    try:
        lower = False
        L = cholesky(A, lower=lower)
        if any(L.diagonal() < CHOLTHRESH):
            raise LinAlgError("Unstable cholesky factor detected")
        X = cho_solve((L, lower), b)
        logdet = cho_log_det(L)

    # Failed cholesky, use svd to do the inverse
    except LinAlgError:

        U, s, V = svd(A)
        X = svd_solve(U, s, V, b)
        logdet = svd_log_det(s)

    return X, logdet


def svd_solve(U, s, V, b, s_tol=1e-15):
    """
    Solve the system :math:`A X = b` for :math:`X`.

    Here :math:`A` is a positive semi-definite matrix using the singular value
    decomposition. This truncates the SVD so only dimensions corresponding to
    non-negative and sufficiently large singular values are used.

    Parameters
    ----------
    U: ndarray
        The :code:`U` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    s: ndarray
        The :code:`s` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    V: ndarray
        The :code:`V` factor of :code:`U, s, V = svd(A)` positive
        semi-definite matrix.
    b: ndarray
        An array or matrix
    s_tol: float
        Cutoff for small singular values. Singular values smaller than
        :code:`s_tol` are clamped to :code:`s_tol`.

    Returns
    -------
    X: ndarray
        The result of :math:`X = A^-1 b`
    okind: ndarray
        The indices of :code:`s` that are kept in the factorisation
    """
    # Test shapes for efficient computations
    n = U.shape[0]
    assert(b.shape[0] == n)
    m = b.shape[1] if np.ndim(b) > 1 else 1

    # Auto clamp SVD based on threshold
    sclamp = np.maximum(s, s_tol)

    # Inversion factors
    ss = 1. / np.sqrt(sclamp)
    U2 = U * ss[np.newaxis, :]
    V2 = ss[:, np.newaxis] * V

    if m < n:
        # Few queries
        X = U2.dot(V2.dot(b))  # O(n^2 (2m))
    else:
        X = U2.dot(V2).dot(b)  # O(n^2 (m + n))

    return X


def hadamard(Y, ordering=True):
    """
    *Very fast* Hadamard transform for single vector at a time.

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

    Examples
    --------
    from https://en.wikipedia.org/wiki/Hadamard_transform with normalisation

    >>> y = np.array([[1, 0, 1, 0, 0, 1, 1, 0]])
    >>> hadamard(y, ordering=False)
    array([[ 0.5 ,  0.25,  0.  , -0.25,  0.  ,  0.25,  0.  ,  0.25]])
    >>> hadamard(y, ordering=True)
    array([[ 0.5 ,  0.  ,  0.  ,  0.  , -0.25,  0.25,  0.25,  0.25]])
    """
    # dot is a sum product over the last axis of a and the second-to-last of b.
    # Transpose - can specify axes, default transposes a[0] and a[1] hmm
    n_vectors, n_Y = Y.shape
    matching = (n_vectors, 2, int(n_Y / 2))
    H = np.array([[1, 1], [1, -1]]) / 2.  # Julia uses 2 and not sqrt(2)?
    steps = int(np.log(n_Y) / np.log(2))
    assert(2**steps == n_Y)  # required
    for _ in range(steps):
        Y = np.transpose(Y.reshape(matching), (0, 2, 1)).dot(H)
    Y = Y.reshape((n_vectors, n_Y))
    if ordering:
        Y = Y[:, _sequency(n_Y)]
    return Y


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
