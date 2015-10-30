"""
Implementation of Bayesian GLMs with nonparametric variational inference [1_].

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
"""


from __future__ import division

import numpy as np
import logging
from scipy.stats.distributions import gamma

from .optimize import minimize, sgd
from .utils import list_to_params as l2p, CatParameters, Positive, Bound, \
    checktypes

# Set up logging
log = logging.getLogger(__name__)


def glm_learn(y, X, likelihood, lparams, basis, bparams, reg=1., postcomp=10,
              maxit=1000, ftol=1e-6, verbose=True):

    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    K = postcomp

    # Intialise m and C
    minit = np.random.randn(D, K)  # TODO OR CLUSTER PHI?
    Cinit = gamma.rvs(0.1, reg / 0.1, size=(D, K))

    # Initial parameter vector
    vparams = [minit, Cinit, reg, lparams, bparams]
    loginds = [1, 2]
    bpos, lpos = False, False
    if checktypes(likelihood.bounds, Positive):
        loginds.append(3)
        lpos = True
    if checktypes(basis.bounds, Positive):
        loginds.append(4)
        bpos = True
    pcat = CatParameters(vparams, log_indices=loginds)

    def L1(mk, m, k, C, _reg, _lparams, _bparams):

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

        # print(k, L1)

        # import IPython; IPython.embed();

        return -L1, -dmk

    def L2(C, m, _reg, _lparams, _bparams):

        C = C.reshape(m.shape)
        Phi = basis(X, *_bparams)  # N x D
        f = Phi.dot(m)  # N x K

        # Posterior responsability terms
        logqkk = qmatrix(m, C)
        logqk = logsumexp(logqkk, axis=1)  # log term of Eq. 7 from [1]

        # Common likelihood calculations
        ll = [likelihood.loglike(y, fk, *_lparams) for fk in f.T]
        d2ll = [likelihood.d2f(y, fk, *_lparams) for fk in f.T]
        H = np.array([(d2llk[:, np.newaxis] * Phi**2).sum(axis=0)
                      for d2llk in d2ll]).T - 1. / _reg

        # Objective, Eq. 10 in [1]
        L2 = 1. / K * (np.sum(ll)
                       - 0.5 * D * K * np.log(2 * np.pi * _reg)
                       - 0.5 * (m**2).sum() / _reg
                       + 0.5 * (C * H).sum()
                       - logqk.sum()) + np.log(K)

        print(L2)

        # Covariance gradients
        pz = np.exp(logqkk - logqk)
        dC = np.zeros_like(C)
        for k in range(K):
            iCkCj = 1 / (C[:, k:k + 1] + C)
            dC[:, k] = - ((m[:, k:k + 1] - m)**2 * iCkCj**2 - iCkCj).dot(pz[k])

        dC = 0.5 / K * (H + dC)

        return -L2, -dC.flatten()

    obj = np.finfo(float).min
    i = 0
    C = np.ones_like(Cinit)
    m = minit

    while i < 3:

        # objo = obj
        # obj = 0

        for k in range(K):
            res = minimize(L1, m[:, k], ftol=ftol, maxiter=100,
                           args=(m, k, C, reg, lparams, bparams),
                           method='L-BFGS-B')
            m[:, k] = res.x
            # obj -= res.fun

        # obj, dC = L2(C, m, reg, lparams, bparams)
        res = minimize(L2, C.flatten(), ftol=ftol, maxiter=100,
                       args=(m, reg, lparams, bparams), method='L-BFGS-B',
                       bounds=[Positive()] * np.prod(C.shape))
        C = np.reshape(res.x, m.shape)

        if verbose:
            log.info("Iter: {}, Objective = {}".format(i, res.fun))

        i += 1
        # print(objo, obj, (objo - obj) / objo)
        # if ((objo - obj) / objo < ftol) or (i >= maxit):
        #     break


    # NOTE: It would be nice if the optimizer knew how to handle Positive
    # bounds when the log trick is used, so we dont have to have this boiler
    # plate...
    # bounds = [Bound()] * (2 * D * postcomp + 1)
    # bounds += [Bound()] * len(likelihood.bounds) if lpos else likelihood.bounds
    # bounds += [Bound()] * len(basis.bounds) if bpos else basis.bounds
    # res = minimize(L2, pcat.flatten(vparams), bounds=bounds, ftol=ftol,
    #                maxiter=maxit, method='L-BFGS-B', backend='scipy')
    # m, C, reg, lparams, bparams = pcat.unflatten(res.x)

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


def logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    if (X.ndim > 1):
        mx = mx[:, np.newaxis]

    return np.log(np.exp(X - mx).sum(axis=axis)) + mx
