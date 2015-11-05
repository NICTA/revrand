"""
Implementation of Bayesian GLMs with nonparametric variational inference [1_].

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
"""


from __future__ import division

import numpy as np
import logging
from scipy.stats.distributions import gamma

from .transforms import logsumexp
from .optimize import minimize
from .utils import CatParameters, Bound

# Set up logging
log = logging.getLogger(__name__)


def glm_learn(y, X, likelihood, lparams, basis, bparams, reg=1., postcomp=10,
              maxit=1000, ftol=1e-6, verbose=True):

    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    K = postcomp

    # Intialise m and C
    m = np.random.randn(D, K)  # TODO OR CLUSTER PHI?
    C = gamma.rvs(1, reg / 1, size=(D, K))

    # Objective function for means, m
    def L1(mk, k, _reg, _lparams, _bparams):

        m[:, k] = mk
        Phi = basis(X, *_bparams)                      # N x D
        fk = Phi.dot(m[:, k])

        # Posterior responsability terms
        logqkk = qmatrix(m, C, k)
        logqk = logsumexp(logqkk)  # log term of Eq. 7 from [1]
        pz = np.exp(logqkk - logqk)

        # Objective sans trace term, Eq. 11 from [1]
        L1 = 1. / K * (likelihood.loglike(y, fk, *_lparams).sum()
                       - D / 2 * np.log(2 * np.pi * _reg)
                       - 0.5 * (m[:, k]**2).sum() / _reg
                       - logqk + np.log(K))

        # Gradient of posterior mean of component k
        dmk = 1. / K * (likelihood.df(y, fk, *_lparams).dot(Phi)
                        - m[:, k] / _reg
                        + (pz * (m[:, k:k + 1] - m)).sum(axis=1))

        return -L1, -dmk

    # Objective function for covariances, C
    # TODO: Incoporate the hyperparameter optimisation in this closure!
    def L2(_C, _reg, _lparams, _bparams):

        _C = pcat.unflatten(_C)[0]
        Phi = basis(X, *_bparams)  # N x D
        f = Phi.dot(m)  # N x K

        # Posterior responsability terms
        logqkk = qmatrix(m, _C)
        logqk = logsumexp(logqkk, axis=1)  # log term of Eq. 7 from [1]

        # Common likelihood calculations
        ll = [likelihood.loglike(y, fk, *_lparams).sum() for fk in f.T]
        d2ll = [likelihood.d2f(y, fk, *_lparams) for fk in f.T]
        H = np.array([(d2llk[:, np.newaxis] * Phi**2).sum(axis=0)
                      for d2llk in d2ll]).T - 1. / _reg

        # Objective, Eq. 10 in [1]
        L2 = 1. / K * (np.sum(ll)
                       - 0.5 * D * K * np.log(2 * np.pi * _reg)
                       - 0.5 * (m**2).sum() / _reg
                       + 0.5 * (_C * H).sum()
                       - logqk.sum()) + np.log(K)

        # Covariance gradients
        pz = np.exp(logqkk - logqk)
        dC = np.zeros_like(_C)
        for k in range(K):
            iCkCj = 1 / (_C[:, k:k + 1] + _C)
            dC[:, k] = - ((m[:, k:k + 1] - m)**2 * iCkCj**2 - iCkCj).dot(pz[k])

        dC = 0.5 / K * (H + dC)

        return -L2, -pcat.flatten_grads([_C], [dC])

    obj = np.finfo(float).min
    pcat = CatParameters([C], log_indices=[0])

    for i in range(maxit):

        objo = obj

        for k in range(K):
            res = minimize(L1, m[:, k], ftol=ftol, maxiter=100,
                           args=(k, reg, lparams, bparams),
                           method='L-BFGS-B')
            m[:, k] = res.x

        res = minimize(L2, pcat.flatten([C]), ftol=ftol, maxiter=100,
                       args=(reg, lparams, bparams), method='L-BFGS-B',
                       bounds=[Bound()] * np.prod(C.shape))
        obj = -res.fun
        C = pcat.unflatten(res.x)[0]

        if verbose:
            log.info("Iter: {}, Objective = {}".format(i, obj))

        if ((objo - obj) / objo < ftol):
            break

    if verbose:
        log.info("Finished! Objective = {}, reg = {}, bparams = {}, "
                 "lparams = {}.".format(obj, reg, bparams, lparams))

    return m, C, reg, lparams, bparams


def glm_predict(Xs, likelihood, basis, m, C, reg, lparams, bparams,
                nsamples=100):

    D, K = m.shape
    w = np.zeros((D, nsamples))

    for k in range(K):
        w += m[:, k:k + 1] + np.random.randn(D, nsamples) \
            * np.sqrt(C[:, k:k + 1])

    w /= K
    f = basis(Xs, *bparams).dot(w)

    return likelihood.Ey(f, *lparams)  # .mean(axis=1)


#
# Private module functions
#

def dgausll(x, mean, dcov):

    D = len(x)
    return - 0.5 * (D * np.log(2 * np.pi) + np.log(dcov).sum()
                    + ((x - mean)**2 / dcov).sum())


def qmatrix(m, C, k=None):

    K = m.shape[1]
    if k is None:
        logq = [[dgausll(m[:, i], m[:, j], C[:, i] + C[:, j])
                 for i in range(K)]
                for j in range(K)]
    else:
        logq = [dgausll(m[:, k], m[:, j], C[:, k] + C[:, j]) for j in range(K)]

    return np.array(logq)
