"""
The standard Bayesian linear regression model.

By using the appropriate bases, this will also yield an implementation of the
"A la Carte" GP [1]_.

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
"""

from __future__ import division

import logging
from functools import partial

import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils import check_random_state

from .utils import atleast_list, issequence
from .mathfun.linalg import solve_posdef
from .optimize import structured_minimizer, logtrick_minimizer
from .btypes import Parameter, Positive
from .basis_functions import apply_grad

# Set up logging
log = logging.getLogger(__name__)


class StandardLinearModel(BaseEstimator, RegressorMixin):
    """
    Bayesian Standard linear model.

    Parameters
    ----------
    basis : Basis
        A basis object, see the basis_functions module.
    var : Parameter, optional
        observation variance initial value.
    tol : float, optional
        optimiser function tolerance convergence criterion.
    maxiter : int, optional
        maximum number of iterations for the optimiser.
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    random_state : None, int or RandomState, optional
        random seed (mainly for random starts)
    """

    def __init__(self,
                 basis,
                 var=Parameter(gamma(1.), Positive()),
                 tol=1e-8,
                 maxiter=1000,
                 nstarts=100,
                 random_state=None
                 ):
        """See class docstring."""
        self.basis = basis
        self.var = var
        self.tol = tol
        self.maxiter = maxiter
        self.nstarts = nstarts
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

    def fit(self, X, y):
        """
        Learn the hyperparameters of a Bayesian linear regressor.

        Parameters
        ----------
        X : ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y : ndarray
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
        X, y = check_X_y(X, y)

        self.obj_ = -np.inf

        # Make list of parameters and decorate optimiser to undestand this
        params = [self.var, self.basis.regularizer, self.basis.params]
        nmin = structured_minimizer(logtrick_minimizer(minimize))

        # Close over objective and learn parameters
        elbo = partial(StandardLinearModel._elbo, self, X, y)
        res = nmin(elbo,
                   params,
                   method='L-BFGS-B',
                   jac=True,
                   tol=self.tol,
                   options={'maxiter': self.maxiter, 'maxcor': 100},
                   random_state=self.random_,
                   nstarts=self.nstarts
                   )

        # Upack learned parameters and report
        self.var_, self.regularizer_, self.hypers_ = res.x

        log.info("Done! ELBO = {}, var = {}, reg = {}, hypers = {}, "
                 "message = {}."
                 .format(-res['fun'],
                         self.var_,
                         self.regularizer_,
                         self.hypers_,
                         res.message)
                 )

        return self

    def _elbo(self, X, y, var, reg, hypers):

        # Get Basis
        Phi = self.basis.transform(X, *atleast_list(hypers))  # N x D
        PhiPhi = Phi.T.dot(Phi)
        N, D = Phi.shape

        # Get regularizer
        L, slices = self.basis.regularizer_diagonal(X, *atleast_list(reg))
        iL = 1. / L

        # Posterior Parameters
        iC = np.diag(iL) + PhiPhi / var
        C, logdetiC = solve_posdef(iC, np.eye(D))
        logdetC = - logdetiC
        m = C.dot(Phi.T.dot(y)) / var

        # Common calcs
        TrPhiPhiC = (PhiPhi * C).sum()
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()

        # Calculate ELBO
        ELBO = -0.5 * (N * np.log(2 * np.pi * var)
                       + sqErr / var
                       + TrPhiPhiC / var
                       + ((m**2 + C.diagonal()) * iL).sum()
                       - logdetC
                       + np.log(L).sum()
                       - D)

        # Cache optimal parameters so we don't have to recompute them later
        if ELBO > self.obj_:
            self.weights_ = m
            self.covariance_ = C
            self.obj_ = ELBO

        log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(ELBO, var, reg, hypers))

        # Grad var
        dvar = 0.5 * (-N + (sqErr + TrPhiPhiC) / var) / var

        # Grad reg
        def dreg(s):
            return - 0.5 * (((m[s]**2 + C[s, s].diagonal()) * iL[s]**2).sum()
                            - iL[s].sum())

        dL = list(map(dreg, slices)) if issequence(slices) else dreg(slices)

        # Get structured basis function gradients
        def dhyps(dPhi):
            return - (m.T.dot(Err.dot(dPhi))
                      - (dPhi.T.dot(Phi) * C).sum()) / var

        dhypers = apply_grad(dhyps, self.basis.grad(X, *atleast_list(hypers)))

        return -ELBO, [-dvar, dL, dhypers]

    def predict(self, X):
        """
        Predict mean from Bayesian linear regression.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, d dimensions).

        Returns
        -------
        Ey : ndarray
            The expected value of y* for the query inputs, X* of shape (N*,).
        """
        Ey, _ = self.predict_moments(X)

        return Ey

    def predict_moments(self, X):
        """
        Full predictive distribution from Bayesian linear regression.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, d dimensions).

        Returns
        -------
        Ey : ndarray
            The expected value of y* for the query inputs, X* of shape (N*,).
        Vy : ndarray
            The expected variance of y* for the query inputs, X* of shape
            (N*,).
        """
        check_is_fitted(self, ['var_', 'regularizer_', 'weights_',
                               'covariance_', 'hypers_'])
        X = check_array(X)

        Phi = self.basis.transform(X, *atleast_list(self.hypers_))
        Ey = Phi.dot(self.weights_)
        Vf = (Phi.dot(self.covariance_) * Phi).sum(axis=1)

        return Ey, Vf + self.var_

    def __repr__(self):
        """Representation."""
        return "{}(basis={}, var={}, tol={}, maxiter={}, nstarts={}, "\
            "random_state={})".format(
                self.__class.__name__,
                self.basis,
                self.var,
                self.tol,
                self.maxiter,
                self.nstarts,
                self.random_state
            )
