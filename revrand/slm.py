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

import logging

import numpy as np
from scipy.optimize import minimize

from .utils import append_or_extend
from .mathfun.linalg import solve_posdef
from .optimize import structured_minimizer, logtrick_minimizer
from .btypes import Parameter, Positive, get_values
from .basis_functions import apply_grad

# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, basis, var=Parameter(1., Positive()),
          regulariser=Parameter(1., Positive()), tol=1e-8, maxiter=1000):
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
        var: Parameter, optional
            observation variance initial value.
        regulariser: Parameter, optional
            weight regulariser (variance) initial value.
        tol: float, optional
            optimiser function tolerance convergence criterion.
        maxiter: int, optional
            maximum number of iterations for the optimiser.

    Returns
    -------
        m: ndarray
            (D,) array of posterior weight means (D is the dimension of the
            features).
        C: ndarray
            (D, D) array of posterior weight variances.
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

        This uses the python logging module for displaying learning status.
        To view these messages have something like,

        .. code ::

            import logging
            logging.basicConfig(level=logging.INFO)
            log = logging.getLogger(__name__)

        in your calling code.
    """

    if y.ndim != 1:
        raise ValueError("y has to be a 1-d array (single task)")
    if X.ndim != 2:
        raise ValueError("X has to be a 2-d array")

    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *get_values(basis.params)).shape[1]

    # Caches for returning optimal params
    ELBOcache = [-np.inf]
    mcache = np.zeros(D)
    Ccache = np.zeros((D, D))

    def ELBO(_var, _lambda, *_theta):

        # Get Basis
        Phi = basis(X, *_theta)                      # N x D
        PhiPhi = Phi.T.dot(Phi)

        # Posterior Parameters
        iC = np.diag(np.ones(D) / _lambda) + PhiPhi / _var
        C, logdetC = solve_posdef(iC, np.eye(D))
        m = C.dot(Phi.T.dot(y)) / _var

        # Common calcs
        TrPhiPhiC = (PhiPhi * C).sum()
        TrC = np.trace(C)
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

        log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(ELBO, _var, _lambda, _theta))

        # Grad var
        dvar = 0.5 * (-N + (sqErr + TrPhiPhiC) / _var) / _var

        # Grad reg
        dlambda = 0.5 * ((TrC + mm) / _lambda - D) / _lambda

        # Get structured basis function gradients
        def dtheta(dPhi):
            return - (m.T.dot(Err.dot(dPhi))
                      - (dPhi.T.dot(Phi) * C).sum()) / _var

        dtheta = apply_grad(dtheta, basis.grad(X, *_theta))

        return -ELBO, append_or_extend([-dvar, -dlambda], dtheta)

    params = append_or_extend([var, regulariser], basis.params)
    nmin = structured_minimizer(logtrick_minimizer(minimize))
    res = nmin(ELBO, params, method='L-BFGS-B', jac=True, tol=tol,
               options={'maxiter': maxiter, 'maxcor': 100})
    (var, regulariser), hypers = res.x[:2], res.x[2:]

    log.info("Done! ELBO = {}, var = {}, reg = {}, hypers = {}, message = {}."
             .format(-res['fun'], var, regulariser, hypers, res.message))

    return mcache, Ccache, hypers, var


def predict(Xs, basis, m, C, hypers, var):
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
            (D, D) array of regression weight covariances (posterior).
        hypers: sequence
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

    Phi_s = basis(Xs, *hypers)

    Ey = Phi_s.dot(m)
    Vf = (Phi_s.dot(C) * Phi_s).sum(axis=1)

    return Ey, Vf, Vf + var
