"""
Various Bayesian linear regression learning and prediction functions.

By using the appropriate bases, this will also yield a simple implementation of
the "A la Carte" GP [1]_.

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
"""

from __future__ import division

import numpy as np
import logging

from scipy.optimize import minimize
from scipy.linalg import cho_solve

from .utils import append_or_extend, safediv
from .linalg import jitchol, cho_log_det
from .optimize import Positive, structured_minimizer, logtrick_minimizer
from .basis_functions import apply_grad

# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, basis, bparams, var=1., regulariser=1., diagcov=False,
          tol=1e-6, maxit=1000, verbose=True):
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
        tol: float, optional
            optimiser function tolerance convergence criterion.
        maxit: int, optional
            maximum number of iterations for the optimiser.

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

    def ELBO(_var, _lambda, *_theta):

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)

        # Safe divide
        ivar = safediv(1., _var)
        ilambda = safediv(1., _lambda)

        # Posterior Parameters
        lower = False
        LiC = jitchol(np.diag(np.ones(D) * ilambda) + PhiPhi * ivar,
                      lower=lower)
        m = cho_solve((LiC, lower), Phi.T.dot(y)) * ivar

        # Common calcs dependent on form of C
        if diagcov:
            C = 1. / (PhiPhi.diagonal() * ivar + ilambda)
            TrPhiPhiC = (PhiPhi.diagonal() * C).sum()
            logdetC = np.log(C).sum()
            TrC = C.sum()
        else:
            C = cho_solve((LiC, lower), np.eye(D))
            TrPhiPhiC = (PhiPhi * C).sum()
            logdetC = -cho_log_det(LiC)
            TrC = np.trace(C)

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        ELBO = -0.5 * (N * np.log(2 * np.pi * _var)
                       + sqErr * ivar
                       + TrPhiPhiC * ivar
                       + (TrC + mm) * ilambda
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

        # Grad var
        dvar = 0.5 * ivar * (-N + (sqErr + TrPhiPhiC) * ivar)

        # Grad reg
        dlambda = 0.5 * ilambda * ((TrC + mm) * ilambda - D)

        # Get structured basis function gradients
        def dtheta(dPhi):
            dPhiPhi = (dPhi * Phi).sum(axis=0) if diagcov else dPhi.T.dot(Phi)
            return - (m.T.dot(Err.dot(dPhi)) - (dPhiPhi * C).sum()) * ivar

        dtheta = apply_grad(dtheta, basis.grad(X, *_theta))

        return -ELBO, append_or_extend([-dvar, -dlambda], dtheta)

    bounds = append_or_extend([Positive(), Positive()], basis.bounds)
    nmin = structured_minimizer(logtrick_minimizer(minimize))
    res = nmin(ELBO, [var, regulariser] + bparams, method='L-BFGS-B', jac=True,
               bounds=bounds, tol=tol, options={'maxiter': maxit})
    (var, regulariser), bparams = res.x[:2], res.x[2:]

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}, "
                 "message = {}."
                 .format(-res['fun'], var, regulariser, bparams, res.message))

    return mcache, Ccache, bparams, var


def predict(Xs, basis, m, C, bparams, var):
    """
    Predict using Bayesian linear regression.

    Parameters
    ----------
        Xs: ndarray
            (Ns,d) array query input dataset (Ns samples, d dimensions).
        basis: Basis
            A basis object, see the basis_functions module.
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
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

    Phi_s = basis(Xs, *bparams)

    Ey = Phi_s.dot(m)
    if C.ndim == 2:
        Vf = (Phi_s.dot(C) * Phi_s).sum(axis=1)
    else:
        Vf = ((Phi_s * C) * Phi_s).sum(axis=1)

    return Ey, Vf, Vf + var
