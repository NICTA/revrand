"""
The standard Bayesian linear regression model.

By using the appropriate bases, this will also yield a simple implementation of
the "A la Carte" GP [1]_.

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
"""

from __future__ import division

import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .utils import append_or_extend
from .mathfun.linalg import solve_posdef
from .optimize import structured_minimizer, logtrick_minimizer
from .btypes import Parameter, Positive
from .basis_functions import apply_grad

# Set up logging
log = logging.getLogger(__name__)


class StandardLinearModel(BaseEstimator, RegressorMixin):
    """
    Standard linear model interface class.

    Parameters
    ----------
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
    """

    def __init__(self,
                 basis,
                 var=Parameter(1., Positive()),
                 regulariser=Parameter(1., Positive()),
                 tol=1e-8,
                 maxiter=1000
                 ):

        self.basis = basis
        self.var = var
        self.regulariser = regulariser
        self.tol = tol
        self.maxiter = maxiter

    def fit(self, X, y):
        """
        Learn the parameters and hyperparameters of a Bayesian linear
        regressor.

        Parameters
        ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)

        Returns
        -------
        self

        Notes
        -----
        This actually optimises the evidence lower bound on log marginal
        likelihood, rather than log marginal likelihood directly. In the case
        of a full posterior convariance matrix, this bound is tight and the
        exact solution will be found (modulo local minima for the
        hyperparameters).

        This uses the python logging module for displaying learning status. To
        view these messages have something like,

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

        self.obj = -np.inf

        # Make list of parameters and decorate optimiser to undestand this
        params = append_or_extend([self.var, self.regulariser],
                                  self.basis.params)
        nmin = structured_minimizer(logtrick_minimizer(minimize))

        # Close over objective and learn parameters
        elbo = partial(StandardLinearModel._elbo, self, X, y)
        res = nmin(elbo,
                   params,
                   method='L-BFGS-B',
                   jac=True, tol=self.tol,
                   options={'maxiter': self.maxiter, 'maxcor': 100}
                   )

        # Upack learned parameters and report
        (self.var, self.regulariser), self.hypers = res.x[:2], res.x[2:]

        log.info("Done! ELBO = {}, var = {}, reg = {}, hypers = {}, "
                 "message = {}."
                 .format(-res['fun'],
                         self.var,
                         self.regulariser,
                         self.hypers,
                         res.message)
                 )

        return self

    def _elbo(self, X, y, var, reg, *hypers):

        # Get Basis
        Phi = self.basis(X, *hypers)                      # N x D
        PhiPhi = Phi.T.dot(Phi)
        N, D = Phi.shape

        # Posterior Parameters
        iC = np.diag(np.ones(D) / reg) + PhiPhi / var
        C, logdetiC = solve_posdef(iC, np.eye(D))
        logdetC = - logdetiC
        m = C.dot(Phi.T.dot(y)) / var

        # Common calcs
        TrPhiPhiC = (PhiPhi * C).sum()
        TrC = np.trace(C)
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        # Calculate ELBO
        ELBO = -0.5 * (N * np.log(2 * np.pi * var)
                       + sqErr / var
                       + TrPhiPhiC / var
                       + (TrC + mm) / reg
                       - logdetC
                       + D * np.log(reg)
                       - D)

        # NOTE: In the above, TriPhiPhiC / var = D - TrC / reg when we
        # analytically solve for C, but we need the trace terms for gradients
        # anyway, so we'll keep them.

        # Cache optimal parameters so we don't have to recompute them later
        if ELBO > self.obj:
            self.weights = m
            self.covariance = C
            self.obj = ELBO

        log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(ELBO, var, reg, hypers))

        # Grad var
        dvar = 0.5 * (-N + (sqErr + TrPhiPhiC) / var) / var

        # Grad reg
        dreg = 0.5 * ((TrC + mm) / reg - D) / reg

        # Get structured basis function gradients
        def dhypers(dPhi):
            return - (m.T.dot(Err.dot(dPhi))
                      - (dPhi.T.dot(Phi) * C).sum()) / var

        dhypers = apply_grad(dhypers, self.basis.grad(X, *hypers))

        return -ELBO, append_or_extend([-dvar, -dreg], dhypers)

    def predict(self, X):
        """
        Predict mean from Bayesian linear regression.

        Parameters
        ----------
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, d dimensions).

        Returns
        -------
        Ey: ndarray
            The expected value of y_star for the query inputs, X_star of shape
            (N_star,).
        """

        Ey, _, _ = self.predict_proba(X)

        return Ey

    def predict_proba(self, X):
        """
        Full predictive distribution from Bayesian linear regression.

        Parameters
        ----------
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, d dimensions).

        Returns
        -------
        Ey: ndarray
            The expected value of y_star for the query inputs, X_star of shape
            (N_star,).
        Vf: ndarray
            The expected variance of f_star for the query inputs, X_star of
            shape (N_star,).
        Vy: ndarray
            The expected variance of y_star for the query inputs, X_star of
            shape (N_star,).
        """

        check_is_fitted(self, ['weights', 'covariance', 'hypers'])

        Phi = self.basis(X, *self.hypers)
        Ey = Phi.dot(self.weights)
        Vf = (Phi.dot(self.covariance) * Phi).sum(axis=1)

        return Ey, Vf, Vf + self.var
