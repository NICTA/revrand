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
from scipy.stats.distributions import gamma
from pyalacarte.linalg import jitchol, logdet
from pyalacarte.minimize import minimize, sgd
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
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    vparams = [var, regulariser, bparams]

    # Caches for returning optimal params
    LMLcache = [-np.inf]
    LCcache = []

    def LML(params):

        _var, _lambda, _theta = l2p(vparams, params)

        # Common computations
        Phi = basis(X, *_theta)                       # N x D
        PhiPhi = Phi.T.dot(Phi)                       # D x D
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

        if LML > LMLcache[0]:
            LMLcache[0] = LML
            LCcache[:] = LC

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
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhi = dPhi.T.dot(Phi)  # D x D
            grad[2+i] = - (np.trace(dPhiPhi) / _var
                           - (dPhiPhi * (iCPhi.dot(Phi))).sum() / _var**2  # !
                           + (yiK.dot(dPhi)).dot(Phi.T).dot(yiK.T)) * _lambda

        return -LML, -grad

    bounds = [var_bounds, regulariser_bounds] + basis.bounds

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = minimize(LML, p2l(vparams), bounds=bounds, method=method, ftol=ftol,
                   xtol=1e-8, maxiter=maxit)

    var, regulariser, bparams = l2p(vparams, res['x'])

    if verbose:
        log.info("Done! LML = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    # Posterior parameters
    C = cho_solve(LCcache, np.eye(D))
    m = C.dot(basis(X, *bparams).T.dot(y)) / var

    return m, C, bparams, var


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
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    vparams = [var, regulariser, bparams]

    # Caches for returning optimal params
    ELBOcache = [-np.inf]
    mcache = np.zeros(D)
    Ccache = np.zeros(D)

    def ELBO(params):

        _var, _lambda, _theta = l2p(vparams, params)

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)

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
            mcache[:] = m
            Ccache[:] = C

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
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhidiag = (dPhi * Phi).sum(axis=0)
            grad[2+i] = (m.T.dot(Err.dot(dPhi)) - (dPhiPhidiag*C).sum()) / _var

        return -ELBO, -grad

    bounds = [var_bounds, regulariser_bounds] + basis.bounds

    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = minimize(ELBO, p2l(vparams), bounds=bounds, method=method, ftol=ftol,
                   xtol=1e-8, maxiter=maxit)

    var, regulariser, bparams = l2p(vparams, res['x'])

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return mcache, Ccache, bparams, var


def bayesreg_sgd(X, y, basis, bparams, var=1, regulariser=1e-3, gtol=1e-1,
                 maxit=1e3, rate=0.5, batchsize=100, verbose=True,
                 var_bounds=(1e-7, None), regulariser_bounds=(1e-7, None),
                 C_bounds=(1e-7, None)):
    """ Learn the parameters and hyperparameters of a Bayesian linear regressor
        using the evidence lower bound (ELBO) on log-marginal likelihood.

        Arguments:
            X: NxD array input dataset (N samples, D dimensions)
            y: N array targets (N samples)
            basis: A basis object, see bases.py
            bparams: A sequence of parameters of the basis object
            var, (float): observation variance initial guess
            regulariser, (float): weight regulariser (variance) initial guess
            gtol, (float): SGD tolerance convergence criterion
            maxit, (int): maximum number of iterations for SGD
            rate, (float): SGD learing rate.
            batchsize, (int): number of observations to use per SGD batch.
            verbose, (float): log learning status
            var_bounds, (tuple): of (lower bound, upper bound) on the variance
                parameter, None for unbounded (though it cannot be <= 0)
            regulariser_bounds, (tuple): of (lower bound, upper bound) on the
                regulariser parameter, None for unbounded (though it cannot be
                <= 0)
            C_bounds, (tuple): of (lower bound, upper bound) on the approximate
                weight posterior parameter, None for unbounded (though it
                cannot be <= 0)

        Returns:
            (tuple): with elements,

                (list): learned sequence of basis object hyperparameters
                (float): learned observation variance
                (float): learned weight regluariser
    """

    N, d = X.shape

    # Initialise parameters
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    minit = np.random.randn(D) * 1e-2
    Cinit = gamma.rvs(0.1, regulariser/0.1, size=D)
    vparams = [minit, Cinit, var, regulariser, bparams]

    # Caches for returning optimal params
    m = np.zeros(D)
    C = np.zeros(D)

    def ELBO(params, data):

        y, X = data[:, 0], data[:, 1:]
        m[:], C[:], _var, _lambda, _theta = l2p(vparams, params)

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D
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

        if verbose:
            log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                     .format(ELBO, _var, _lambda, _theta))

        # Mean gradient
        gm = Err.dot(Phi) / _var - m / _lambda

        # Covariance gradient
        gC = 0.5 * (- PPdiag / _var - 1./_lambda + 1./C)

        # Grad var
        gvar = 0.5 * (- N / _var + (sqErr + TrPhiPhiC) / _var**2)

        # Grad reg
        greg = 0.5 / _lambda * ((C.sum() + mm) / _lambda - D)

        # Loop through basis param grads
        gtheta = []
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for i,  dPhi in enumerate(dPhis):
            dPhiPhidiag = (dPhi * Phi).sum(axis=0)
            gt = (m.T.dot(Err.dot(dPhi)) - (dPhiPhidiag*C).sum()) / _var
            gtheta.append(gt)

        return -ELBO, -np.array(p2l([gm, gC, gvar, greg, gtheta]))

    bounds = [(None, None)]*D + [(1e-7, None)]*D + \
        [var_bounds, regulariser_bounds] + basis.bounds
    res = sgd(ELBO, p2l(vparams), np.hstack((y[:, np.newaxis], X)), rate=rate,
              bounds=bounds, gtol=gtol, maxiter=maxit, batchsize=batchsize,
              eval_obj=True)

    _, _, var, regulariser, bparams = l2p(vparams, res['x'])

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        log.info('Termination condition: {}.'.format(res['message']))

    return m, C, bparams, var


def bayesreg_predict(X_star, basis, m, C, bparams, var):
    """ Predict using Bayesian linear regression.

        Arguments:
            X_star: (N_star,D) array query input dataset (N_star samples,
                D dimensions)
            m: (D,) array of regression weights (posterior)
            C: (D,) or (D, D) array of regression weight covaariances
               (posterior)
            basis: A basis object, see bases.py
            bparams: A sequence of hyperparameters of the basis object
            var: observation variance

        Returns:
            array: The expected value of y_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of f_star for the query inputs, X_star
               of shape (N_star,)
            array: The expected variance of y_star for the query inputs, X_star
               of shape (N_star,)
    """

    Phi_s = basis(X_star, *bparams)

    Ey = Phi_s.dot(m)
    if C.ndim == 2:
        Vf = (Phi_s.dot(C) * Phi_s).sum(axis=1)
    else:
        Vf = ((Phi_s * C) * Phi_s).sum(axis=1)

    return Ey, Vf, Vf + var
