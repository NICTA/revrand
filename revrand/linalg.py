""" Various linear algebra utilities. """


from __future__ import division

import numpy as np
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
    LinAlgError: Exceeded maximum jitter limit, yet a is still not positive semidefinite!
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

# deprecated
def logdet(L):
    """ Compute the log determinant of a matrix.

        Arguments:
            L: The [NxN] cholesky factor of the matrix.

        Returns:
            The log determinant (scalar)
    """
    return cho_log_det(L)
