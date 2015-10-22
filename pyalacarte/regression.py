"""
Various Bayesian linear regression learning and prediction functions.

By using the appropriate bases, this will also yield a simple implementation of
the "A la Carte" GP [1]_.

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
"""

# TODO:
#   - Make reshaping dtheta less clunky with log trick

from __future__ import division

import numpy as np
import logging

from scipy.linalg import cho_solve
from scipy.stats.distributions import gamma

from .linalg import jitchol, logdet
from .optimize import minimize, sgd
from .utils import list_to_params as l2p, CatParameters, Positive, Bound, \
    checktypes

# Set up logging
log = logging.getLogger(__name__)


def bayeslinear(X, y, basis, bparams, var=1., regulariser=1., diagcov=False,
                ftol=1e-6, maxit=1000, verbose=True, usegradients=True):
    """
    Learn the parameters and hyperparameters of a Bayesian linear regressor.

    Parameters
    ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        var: float, optional
            observation variance initial value.
        regulariser: float, optional
            weight regulariser (variance) initial value.
        diagcov: bool, optional
            approximate posterior covariance with diagional matrix.
        verbose: bool, optional
            log learning status.
        ftol: float, optional
            optimiser function tolerance convergence criterion.
        maxit: int, optional
            maximum number of iterations for the optimiser.
        usegradients: bool, optional
            True for using gradients to optimize the parameters, otherwise
            false uses BOBYQA (from nlopt).

    Returns
    -------
        m: ndarray
            (D,) array of posterior weight means (D is the dimension of the
            features).
        C: ndarray
            (D,) array of posterior weight variances.
        bparams: sequence
            learned sequence of basis object hyperparameters.
        float:
            learned observation variance

    Notes
    -----
        This actually optimises the evidence lower bound on log marginal
        likelihood, rather than log marginal likelihood directly. In the case
        of a full posterior convariance matrix, this bound is tight and the
        exact solution will be found (modulo local minima for the
        hyperparameters).
    """

    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]

    # Caches for returning optimal params
    ELBOcache = [-np.inf]
    mcache = np.zeros(D)
    Ccache = np.zeros(D) if diagcov else np.zeros((D, D))

    # Initial parameter vector
    vparams = [var, regulariser, bparams]
    posbounds = checktypes(basis.bounds, Positive)
    pcat = CatParameters(vparams, log_indices=[0, 1, 2] if posbounds
                         else [0, 1])

    def ELBO(params):

        uparams = pcat.unflatten(params)
        _var, _lambda, _theta = uparams

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)

        # Posterior Parameters
        LfullC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var), False
        m = cho_solve(LfullC, Phi.T.dot(y)) / _var

        # Common calcs dependent on form of C
        if diagcov:
            C = 1. / (PhiPhi.diagonal() / _var + 1. / _lambda)
            TrPhiPhiC = (PhiPhi.diagonal() * C).sum()
            logdetC = np.log(C).sum()
            TrC = C.sum()
        else:
            C = cho_solve(LfullC, np.eye(D))
            TrPhiPhiC = (PhiPhi * C).sum()
            logdetC = -logdet(LfullC[0])
            TrC = np.trace(C)

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        ELBO = -0.5 * (N * np.log(2 * np.pi * _var)
                       + sqErr / _var
                       + TrPhiPhiC / _var
                       + (TrC + mm) / _lambda
                       - logdetC
                       + D * np.log(_lambda)
                       - D)

        # NOTE: In the above, TriPhiPhiC / _var = D - TrC / _lambda when we
        # analytically solve for C, but we need the trace terms for gradients
        # anyway, so we'll keep them.

        # Cache square error to compute corrected variance
        if ELBO > ELBOcache[0]:
            mcache[:] = m
            Ccache[:] = C
            ELBOcache[0] = ELBO

        if verbose:
            log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                     .format(ELBO, _var, _lambda, _theta))

        if not usegradients:
            return -ELBO

        # Grad var
        dvar = 0.5 / _var * (-N + (sqErr + TrPhiPhiC) / _var)

        # Grad reg
        dlambda = 0.5 / _lambda * ((TrC + mm) / _lambda - D)

        # Loop through basis param grads
        dtheta = []
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for dPhi in dPhis:
            dPhiPhi = (dPhi * Phi).sum(axis=0) if diagcov else dPhi.T.dot(Phi)
            dt = (m.T.dot(Err.dot(dPhi)) - (dPhiPhi * C).sum()) / _var
            dtheta.append(dt)

        # Reconstruct dtheta in shape of theta, NOTE: this is a bit clunky!
        dtheta = l2p(_theta, dtheta)

        return -ELBO, -pcat.flatten_grads(uparams, [dvar, dlambda, dtheta])

    # NOTE: It would be nice if the optimizer knew how to handle Positive
    # bounds when the log trick is used, so we dont have to have this boiler
    # plate...
    bounds = [Bound()] * 2
    bounds += [Bound()] * len(basis.bounds) if posbounds else basis.bounds
    method = 'L-BFGS-B' if usegradients else None  # else BOBYQA numerical
    res = minimize(ELBO, pcat.flatten(vparams), method=method, jac=True,
                   bounds=bounds, ftol=ftol, xtol=1e-8, maxiter=maxit)

    var, regulariser, bparams = pcat.unflatten(res['x'])

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        if not res['success']:
            log.info('Terminated unsuccessfully: {}.'.format(res['message']))

    return mcache, Ccache, bparams, var


def bayeslinear_sgd(X, y, basis, bparams, var=1, regulariser=1., gtol=1e-3,
                    passes=100, rate=0.9, eta=1e-6, batchsize=100,
                    verbose=True):
    """
    Learn the parameters and hyperparameters of an approximate Bayesian linear
    regressor using stochastic gradient descent for large scale problems.


    Parameters
    ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        var: float, optional
            observation variance initial value.
        regulariser: float, optional
            weight regulariser (variance) initial value.
        gtol: float,
            SGD tolerance convergence criterion.
        passes: int, optional
            Number of complete passes through the data before optimization
            terminates (unless it converges first).
        rate: float, optional
            SGD decay rate, must be [0, 1].
        batchsize: int, optional
            number of observations to use per SGD batch.

        verbose, (float): log learning status

    Returns
    -------
        m: ndarray
            (D,) array of posterior weight means (D is the dimension of the
            features).
        C: ndarray
            (D,) array of posterior weight variances.
        bparams: sequence
            learned sequence of basis object hyperparameters.
        float:
            learned observation variance

    Notes
    -----
        This approximates the posterior over the weights with a diagonal
        covariance to keep the parameter count small for SGD. As a consequence,
        features with large dimensionality can be used.
    """

    N, d = X.shape

    # Initialise parameters
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    minit = np.random.randn(D)
    Cinit = gamma.rvs(0.1, regulariser / 0.1, size=D)

    # Initial parameter vector
    vparams = [minit, Cinit, var, regulariser, bparams]
    posbounds = checktypes(basis.bounds, Positive)
    pcat = CatParameters(vparams, log_indices=[1, 2, 3, 4] if posbounds
                         else [1, 2, 3])

    def ELBO(params, data):

        y, X = data[:, 0], data[:, 1:]
        uparams = pcat.unflatten(params)
        m, C, _var, _lambda, _theta = uparams

        # Get Basis
        Phi = basis(X, *_theta)                      # Nb x D
        PPdiag = (Phi**2).sum(axis=0)

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        Nb = len(y)
        TrPhiPhiC = (PPdiag * C).sum()
        ELBO = -0.5 * (Nb * np.log(2 * np.pi * _var)
                       + sqErr / _var
                       + TrPhiPhiC / _var
                       + Nb / N * (
                           + (C.sum() + mm) / _lambda
                           - np.log(C).sum()
                           + D * np.log(_lambda)
                           - D))

        if verbose:
            log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                     .format(ELBO, _var, _lambda, _theta))

        # Mean gradient
        dm = Err.dot(Phi) / _var - m * Nb / (_lambda * N)

        # Covariance gradient
        dC = - 0.5 * (PPdiag / _var + Nb / N * (1. / _lambda - 1. / C))

        # Grad variance
        dvar = 0.5 / _var * (-Nb + (TrPhiPhiC + sqErr) / _var)

        # Grad reg
        dlambda = 0.5 * Nb / (_lambda * N) * ((C.sum() + mm) / _lambda - D)

        # Loop through basis param grads
        dtheta = []
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for dPhi in dPhis:
            dPhiPhidiag = (dPhi * Phi).sum(axis=0)
            dt = (m.T.dot(Err.dot(dPhi)) - (dPhiPhidiag * C).sum()) / _var
            dtheta.append(dt)

        # Reconstruct dtheta in shape of theta, NOTE: this is a bit clunky!
        dtheta = l2p(_theta, dtheta)

        return -ELBO, -pcat.flatten_grads(uparams, [dm, dC, dvar, dlambda,
                                                    dtheta])

    # NOTE: It would be nice if the optimizer knew how to handle Positive
    # bounds when the log trick is used, so we dont have to have this boiler
    # plate...
    bounds = [Bound()] * (2 * D + 2)
    bounds += [Bound()] * len(basis.bounds) if posbounds else basis.bounds
    res = sgd(ELBO, pcat.flatten(vparams), np.hstack((y[:, np.newaxis], X)),
              rate=rate, eta=eta, bounds=bounds, gtol=gtol, passes=passes,
              batchsize=batchsize, eval_obj=True)

    m, C, var, regulariser, bparams = pcat.unflatten(res['x'])

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        log.info('Termination condition: {}.'.format(res['message']))

    return m, C, bparams, var


def bayeslinear_predict(X_star, basis, m, C, bparams, var):
    """
    Predict using Bayesian linear regression.

    Parameters
    ----------
        X_star: ndarray
            (N_star,D) array query input dataset (N_star samples, D
            dimensions).
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of hyperparameters of the basis object.
        var: float
            observation variance.

    Returns
    -------
        Ey: ndarray
            The expected value of y_star for the query inputs, X_star
            of shape (N_star,).
        Vf: ndarray
            The expected variance of f_star for the query inputs,
            X_star of shape (N_star,).
        Vy: ndarray
            The expected variance of y_star for the query inputs,
            X_star of shape (N_star,).
    """

    Phi_s = basis(X_star, *bparams)

    Ey = Phi_s.dot(m)
    if C.ndim == 2:
        Vf = (Phi_s.dot(C) * Phi_s).sum(axis=1)
    else:
        Vf = ((Phi_s * C) * Phi_s).sum(axis=1)

    return Ey, Vf, Vf + var
