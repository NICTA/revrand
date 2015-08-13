""" Various Bayesian linear regression learning and prediction functions.

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
from scipy.linalg import cho_solve
from pyalacarte.linalg import jitchol, logdet
from pyalacarte.minimize import minimize as nmin
from pyalacarte.bases import params_to_list as p2l
from pyalacarte.bases import list_to_params as l2p


# Set up logging
log = logging.getLogger(__name__)


def bayesreg_lml(X, y, basis, bparams, var=1, regulariser=1e-3, ftol=1e-5,
                 maxit=1000, verbose=True, var_bounds=(1e-7, None),
                 regulariser_bounds=(1e-7, None), usegradients=True):
    """ Learn the parameters and hyperparameters of a Bayesian linear regressor
        using log-marginal likelihood.

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
            usegradients: True for using gradients to optimize the parameters,
                otherwise false uses BOBYQA (from nlopt)

        Returns:
            list: learned sequence of basis object hyperparameters
            float: learned observation variance
            float: learned weight regluariser
    """

    N, d = X.shape
    vparams = [var, regulariser, p2l(bparams)]

    def LML(params):

        _var, _lambda, _theta = l2p(vparams, params)

        # Common computations
        Phi = basis.from_vector(X, _theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)                                 # D x D
        D = Phi.shape[1]
        LC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var)
        iCPhi = cho_solve(LC, Phi.T)                            # D x N
        yiK = y.T / _var - (y.T.dot(Phi)).dot(iCPhi) / _var**2  # 1 x N

        # Objective
        LML = -0.5 * (N * np.log(2*np.pi*_var**2)
                      + D * np.log(_lambda)
                      + logdet(LC[0])
                      + yiK.dot(y))

        if verbose:
            log.info("LML = {}, var = {}, reg = {}, bparams = {}."
                     .format(LML, _var, _lambda, _theta))

        if not usegradients:
            return -LML

        # Gradients
        grad = np.empty(len(params))

        # Grad var
        grad[0] = - N / _var + (Phi * iCPhi.T).sum() / (2 * _var**2) \
            + (yiK**2).sum() / 2

        # Grad reg -- the second trace here is the largest computation
        grad[1] = 0.5 * (- np.trace(PhiPhi) / _var
                         + (PhiPhi * (iCPhi).dot(Phi)).sum() / _var**2
                         + (yiK.dot(Phi)**2).sum())

        # Loop through basis param grads
        dPhis = basis.grad_from_vector(X, _theta) if _theta else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhi = dPhi.T.dot(Phi)  # D x D
            grad[2+i] = - (np.trace(dPhiPhi) / _var
                           - (dPhiPhi * (iCPhi.dot(Phi))).sum() / _var**2  # !
                           + (yiK.dot(dPhi)).dot(Phi.T).dot(yiK.T)) * _lambda

        return -LML, -grad

    bounds = [var_bounds, regulariser_bounds] + basis.bounds

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = nmin(LML, p2l(vparams), bounds=bounds, method=method, ftol=ftol,
               xtol=1e-8, maxiter=maxit)

    var, regulariser, bparams = l2p(vparams, res['x'])

    if verbose:
        log.info("Done! LML = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return bparams, var, regulariser


def bayesreg_elbo(X, y, basis, bparams, var=1, regulariser=1e-3, ftol=1e-5,
                  maxit=1000, verbose=True, var_bounds=(1e-7, None),
                  regulariser_bounds=(1e-7, None), usegradients=True):
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
            usegradients: True for using gradients to optimize the parameters,
                otherwise false uses BOBYQA (from nlopt)

        Returns:
            list: learned sequence of basis object hyperparameters
            float: learned observation variance
            float: learned weight regluariser
    """

    N, d = X.shape

    # Caches for correcting the true variance
    sqErrcache = [0]
    ELBOcache = [-np.inf]
    vparams = [var, regulariser, p2l(bparams)]

    def ELBO(params):

        _var, _lambda, _theta = l2p(vparams, params)

        # Get Basis
        Phi = basis.from_vector(X, _theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)
        D = Phi.shape[1]

        # Posterior Parameters
        LfullC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var)
        m = cho_solve(LfullC, Phi.T.dot(y)) / _var
        C = 1. / (PhiPhi.diagonal() / _var + 1. / _lambda)

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        TrPhiPhiC = (PhiPhi.diagonal() * C).sum()
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

        if not usegradients:
            return -ELBO

        # Gradients
        grad = np.empty(len(params))

        # Grad var
        grad[0] = 0.5 * (-N / _var + (sqErr + TrPhiPhiC) / _var**2)

        # Grad reg
        grad[1] = 0.5 / _lambda * ((C.sum() + mm) / _lambda - D)

        # Loop through basis param grads
        dPhis = basis.grad_from_vector(X, _theta) if _theta else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhidiag = (dPhi * Phi).sum(axis=0)
            grad[2+i] = (m.T.dot(Err.dot(dPhi)) - (dPhiPhidiag*C).sum()) / _var

        return -ELBO, -grad

    bounds = [var_bounds, regulariser_bounds] + basis.bounds

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = nmin(ELBO, p2l(vparams), bounds=bounds, method=method, ftol=ftol,
               xtol=1e-8, maxiter=maxit)

    _, regulariser, bparams = l2p(vparams, res['x'])
    var = sqErrcache[0] / (N - 1)  # for corrected

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return bparams, var, regulariser


def bayesreg_predict(X_star, X, y, basis, bparams, var, regulariser):
    """ Predict using Bayesian linear regression.

        Arguments:
            X_star: N_starxD array query input dataset (N_star samples,
                D dimensions)
            X: NxD array training input dataset (N samples, D dimensions)
            y: N array training targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of hyperparameters of the basis object
            var: observation variance
            regulariser: weight regulariser (variance)

        Returns:
            array: The expected value of y_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of f_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of y_star for the query inputs, X_star
               of shape (N_star,)
    """

    Phi = basis(X, *bparams)
    Phi_s = basis(X_star, *bparams)
    N, D = Phi.shape

    LC = jitchol(np.diag(np.ones(D) / regulariser) + Phi.T.dot(Phi) / var)

    Ey = Phi_s.dot(cho_solve(LC, Phi.T.dot(y))) / var
    Vf = (Phi_s * cho_solve(LC, Phi_s.T).T).sum(axis=1)

    return Ey, Vf, Vf + var
