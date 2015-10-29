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

    # Intialise m and C
    minit = np.random.randn(D, postcomp)  # TODO OR CLUSTER PHI?
    Cinit = gamma.rvs(0.1, reg / 0.1, size=(D, postcomp))

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

    def L1(params):
        # TODO argmax L1 w.r.t m_k i.e. for each component
        # NOTE: Test this first independent of the covariances

        pass

    def L2(params):

        uparams = pcat.unflatten(params)
        _reg, _lparams, _bparams = uparams

        # Get Basis
        Phi = basis(X, *_bparams)                      # N x D

        # Common calculations
        logqk = qmatrix(m, C)
        logq = logsumexp(logqk, axis=1)
        pz = np.exp(logqk - logq)

        if verbose:
            log.info("Objective = {}, reg = {}, bparams = {}, lparams = {},"
                     " MAP success: {}"
                     .format(L2, _reg, _bparams, _lparams, res.success))

        return -L2

    # NOTE: It would be nice if the optimizer knew how to handle Positive
    # bounds when the log trick is used, so we dont have to have this boiler
    # plate...
    bounds = [Bound()] * (2 * D * postcomp + 1)
    bounds += [Bound()] * len(likelihood.bounds) if lpos else likelihood.bounds
    bounds += [Bound()] * len(basis.bounds) if bpos else basis.bounds
    res = minimize(L2, pcat.flatten(vparams), bounds=bounds, ftol=ftol,
                   maxiter=maxit, method='L-BFGS-B', backend='scipy')
    m, C, reg, lparams, bparams = pcat.unflatten(res.x)

    if verbose:
        log.info("Finished! LML = {}, reg = {}, bparams = {}, lparams = {},"
                 " Status: {}"
                 .format(-res.fun, reg, bparams, lparams, res.message))

    return m, C, reg, lparams, bparams


#
# Private module functions
#

def dgausll(x, mean, dcov):

    D = len(x)
    return - 0.5 * (D * np.log(2 * np.pi) + np.log(dcov).sum()
                    + ((x - mean)**2 / dcov).sum())


def qmatrix(m, C):

    K = m.shape[1]
    logq = [[dgausll(m[:, k], m[:, j], C[:, k] + C[:, j]) for k in range(K)]
            for j in range(K)]
    return np.array(logq)


def logsumexp(X, axis=0):
    """ Log-sum-exp trick for matrix X for summation along a specified axis """

    mx = X.max(axis=axis)
    return np.log(np.exp(X - mx[:, np.newaxis]).sum(axis=axis)) + mx
