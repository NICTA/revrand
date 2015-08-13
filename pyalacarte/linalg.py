""" Various linear algebra utilities. """


from __future__ import division

import numpy as np
import scipy.linalg as la


def jitchol(A, lower=False):
    """ Do cholesky decomposition with a bit of diagonal jitter if needs be.

        Aarguments:
            A: a [NxN] positive definite symmetric matrix to be decomposed as
                A = L.dot(L.T).
            lower: Return lower triangular factor, default False (upper).

        Returns:
            An upper or lower triangular matrix factor, L, also [NxN].
            Also wheter or not a the matrix is lower triangular form,
            (L, lower).
    """

    # Try the cholesky first
    try:
        cholA = la.cholesky(A, lower=lower)
        return cholA, lower
    except la.LinAlgError:
        pass

    # Now add jitter
    D = A.shape[0]
    jit = 1e-16
    cholA = None
    di = np.diag_indices(D)
    Amean = A.diagonal().mean()

    while jit < 1e-3:

        try:
            Ajit = A.copy()
            Ajit[di] += Amean * jit
            cholA = la.cholesky(Ajit, lower=lower)
            break
        except la.LinAlgError:
            jit *= 10

    if cholA is None:
        raise la.LinAlgError("Added maximum jitter and A still not PSD!")

    return cholA, lower


def logdet(L):
    """ Compute the log determinant of a matrix.

        Arguments:
            L: The [NxN] cholesky factor of the matrix.

        Returns:
            The log determinant (scalar)
    """

    return 2 * np.log(L.diagonal()).sum()
