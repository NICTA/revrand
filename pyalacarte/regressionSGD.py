""" Stochastic Gradient Descent Bayesian Linear Regression using the Evidence
    Lower Bound.

    By using the appropriate bases, this will also yeild a simple
    implementation of the "A la Carte" GP [1].

    References:
        - Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
          Learning Fast Kernels". Proceedings of the Eighteenth International
          Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
          2015.
"""

from __future__ import division

import numpy as np
import logging
from scipy.stats.distributions import gamma
from pyalacarte.minimize import minimize as nmin
from pyalacarte.bases import params_to_list as p2l
from pyalacarte.bases import list_to_params as l2p

# Set up logging
log = logging.getLogger(__name__)


def bayesreg_sgd(X, y, basis, bparams, var=1, regulariser=1e-3, ftol=1e-7,
                 maxit=1000, verbose=True, var_bounds=(1e-7, None),
                 regulariser_bounds=(1e-7, None)):
    """ Learn the parameters and hyperparameters of a Bayesian linear regressor
        using the evidence lower bound (ELBO) on log-marginal likelihood.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of parameters of the basis object
            var: observation variance initial guess
            regulariser: weight regulariser (variance) initial guess
            verbose: log learning status
            ftol: optimiser function tolerance convergence criterion
            maxit: maximum number of iterations for the optimiser
            var_bounds: tuple of (lower bound, upper bound) on the variance
                parameter, None for unbounded (though it cannot be <= 0)
            regulariser_bounds: tuple of (lower bound, upper bound) on the
                regulariser parameter, None for unbounded (though it cannot be
                <= 0)

        Returns:
            list: learned sequence of basis object hyperparameters
            float: learned observation variance
            float: learned weight regluariser
    """

    N, d = X.shape

    # Caches for correcting the true variance
    sqErrcache = [0]
    ELBOcache = [-np.inf]

    # Initialise parameters
    Phi = basis(X, *bparams)
    D = Phi.shape[1]
    minit = np.linalg.solve(np.diag(np.ones(D)/regulariser)+Phi.T.dot(Phi)/var,
                            Phi.T.dot(y) / var)
    Cinit = gamma.rvs(0.1, regulariser/0.1, size=D)
    vparams = [minit, Cinit, var, regulariser, p2l(bparams)]

    def ELBO(params):

        m, C, _var, _lambda, _theta = l2p(vparams, params)

        # Get Basis
        Phi = basis.from_vector(X, _theta)                      # N x D
        PPdiag = (Phi**2).sum(axis=0)

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        TrPhiPhiC = (PPdiag * C).sum()
        ELBO = -0.5 * (N * np.log(2 * np.pi * _var)
                       + sqErr / _var
                       + TrPhiPhiC / _var
                       + C.sum() / _lambda
                       - np.log(C).sum()
                       + mm / _lambda
                       + D * np.log(_lambda)
                       - D)

        # Cache square error to compute corrected variance
        if ELBO > ELBOcache[0]:
            sqErrcache[0] = sqErr

        if verbose:
            log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                     .format(ELBO, _var, _lambda, _theta))

        # Mean gradient
        gm = Err.dot(Phi) / _var - m / _lambda

        # Covariance gradient
        gC = 0.5 * (- PPdiag / _var - 1./_lambda + 1./C)

        # Grad var
        gvar = 0.5 * (-N / _var + (sqErr + TrPhiPhiC) / _var**2)

        # Grad reg
        greg = 0.5 / _lambda * ((C.sum() + mm) / _lambda - D)

        # Loop through basis param grads
        gtheta = []
        dPhis = basis.grad_from_vector(X, _theta) if _theta else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhidiag = (dPhi * Phi).sum(axis=0)
            gt = (m.T.dot(Err.dot(dPhi)) - (dPhiPhidiag*C).sum()) / _var
            gtheta.append(gt)

        return -ELBO, -np.array(p2l([gm, gC, gvar, greg, gtheta]))

    bounds = [(None, None)]*D + [(1e-7, None)]*D + \
        [var_bounds, regulariser_bounds] + basis.bounds

    res = nmin(ELBO, p2l(vparams), bounds=bounds, method='L-BFGS-B', ftol=ftol,
               xtol=1e-8, maxiter=maxit)

    _, _, _, regulariser, bparams = l2p(vparams, res['x'])
    var = sqErrcache[0] / (N - 1)  # for corrected, otherwise res['x'][2]

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return bparams, var, regulariser
