"""Various linear algebra utilities."""


from __future__ import division

import autograd.numpy as np
from autograd.scipy.linalg import solve_triangular


def jitchol(a, jit=None, jit_max=1e-3, returns_jit=False):
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
    >>> jitchol(a)
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
    array([[ 1.41,  0.  ,  0.  ],
           [-0.71,  1.22,  0.  ],
           [ 0.  , -0.82,  1.15]])
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
        chol = np.linalg.cholesky(a)
        if returns_jit:
            return chol, jit
        else:
            return chol

    except np.linalg.LinAlgError:

        if jit is None:
            jit = 1e-16

        if jit > jit_max:
            raise np.linalg.LinAlgError('Exceeded maximum jitter limit, yet a is still'
                                        ' not positive semidefinite!')

        diag = np.diag(a)
        diag_mean = np.mean(diag)
        diag_delta = jit * diag_mean

        return jitchol(a + diag_delta * np.eye(*a.shape), jit=10 * jit,
                       jit_max=jit_max, returns_jit=returns_jit)


def cho_solve(c_and_lower, b):
    """
    Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cholesky
    b : array
        Right-hand side
    Returns
    -------
    x : array
        The solution to the system A x = b
    Examples
    --------
    >>> import scipy as sp
    >>> c = np.random.randn(4, 4)
    >>> a = c.dot(c.T)
    >>> a_inv = np.linalg.solve(a, np.identity(4))

    Lower Cholesky Decomposition

    >>> l = sp.linalg.cholesky(a, lower=True)
    >>> np.allclose(l.dot(l.T), a)
    True
    >>> np.allclose(cho_solve((l, True), np.identity(4)), a_inv)
    True
    >>> np.allclose(cho_solve((l, True), np.identity(4)),
    ...             sp.linalg.cho_solve((l, True), np.identity(4)))
    True

    Upper Cholesky Decomposition

    >>> u = sp.linalg.cholesky(a, lower=False)
    >>> np.allclose(u.T.dot(u), a)
    True
    >>> np.allclose(cho_solve((u, False), np.identity(4)), a_inv)
    True
    >>> np.allclose(cho_solve((u, False), np.identity(4)),
    ...             sp.linalg.cho_solve((u, False), np.identity(4)))
    True
    """
    c, lower = c_and_lower

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("The factored matrix c is not square.")

    if c.shape[1] != b.shape[0]:
        raise ValueError("incompatible dimensions.")

    if not lower:
        c = c.T

    y = solve_triangular(c, b, trans='N', lower=True)
    x = solve_triangular(c, y, trans='T', lower=True)

    return x


def cho_log_det(c):
    """
    Compute the log of the determinant of `A`, given its (upper or lower)
    Cholesky factorization `c`.

    Examples
    --------
    >>> from scipy.linalg import cholesky
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
    return 2 * np.sum(np.log(np.diag(c)))
