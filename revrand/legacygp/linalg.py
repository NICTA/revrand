import numpy as np


def svd_inverse(factorisation):
    R, s, Rt = factorisation
    # R, S, RT = np.linalg.svd(K)
    ss = 1./np.sqrt(s)
    R2 = R * ss[np.newaxis, :]
    return R2.dot(R2.T)
    # log determinant is product of s


def svd_yKy(factorisation, y):
    # efficiently computes y' Kinv y
    if y.ndim < 2:
        y = np.atleast_2d(y).T  # view... dot behaves strangely otherwise
    R, s, Rt = factorisation
    ss = 1./np.sqrt(s)
    # K = R.dot(np.diag(s).dot(Rt))
    # true = y.T.dot(np.linalg.inv(K).dot(y))
    F = ss[:, np.newaxis] * np.linalg.solve(R, y)  # checked!
    return np.sum(F**2)


def svd_log_det(factorisation):
    """ Computes log determinant for a svd factorisation
    """
    R, s, Rt = factorisation
    return np.sum(np.log(s))


def svd_solve(factorisation, M):
    """ Computes A\M for factorisation = np.linalg.svd(A)
    """
    R, s, Rt = factorisation
    n = R.shape[0]
    assert(M.shape[0] == n)
    m = M.shape[1] if np.ndim(M) > 1 else 1
    ss = 1./np.sqrt(s)
    R2 = R * ss[np.newaxis, :]

    if m < n:
        # Few queries
        return R2.dot(R2.T.dot(M))  # O(n^2 (2m))
    else:
        return R2.dot(R2.T).dot(M)  # O(n^2 (m + n))


def svd_half_solve(factorisation, M):
    """ Computes A\M for factorisation = np.linalg.svd(A)
    """
    R, s, Rt = factorisation
    ss = 1./np.sqrt(s)
    R2 = R * ss[np.newaxis, :]
    return R2.T.dot(M)  # O(n^2 (2m))
