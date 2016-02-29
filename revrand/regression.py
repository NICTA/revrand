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

import autograd.numpy as np
import logging

from scipy.stats.distributions import gamma

from .linalg import (cho_solve, jitchol, cho_log_det)
from .optimize import (sgd, Bound, Positive, structured_sgd, logtrick_sgd,
                       decorated_minimize as minimize)

from .utils.base import Bunch
from .utils.autograd import value_and_multigrad
from .types.functions import FuncRes, func_negate, func_value
from .basis_functions import identity

# Set up logging
log = logging.getLogger(__name__)


def make_elbo(X, y, basis_func, cache=Bunch()):

    N, d = X.shape

    def elbo(var, lambda_, *thetas):

        Phi = basis_func(X, *thetas).value
        PhiPhi = np.dot(Phi.T, Phi)

        _, D = Phi.shape

        LiC = jitchol(np.diag(np.ones(D) / lambda_) + PhiPhi / var)

        m = cho_solve((LiC, True), np.dot(Phi.T, y)) / var
        C = cho_solve((LiC, True), np.eye(D))

        TrPhiPhiC = np.sum(PhiPhi * C)
        logdetC = - cho_log_det(LiC)
        TrC = np.trace(C)

        Err = y - np.dot(Phi, m)
        sqErr = np.sum(Err**2)
        mm = np.sum(m**2)

        # Function value
        elb = -.5 * (N * np.log(2. * np.pi * var)
                     + sqErr / var
                     + TrPhiPhiC / var
                     + (TrC + mm) / lambda_
                     - logdetC
                     + D * np.log(lambda_)
                     - D)

        cache.elb = elb
        cache.m = m
        cache.C = C

        # Gradient computations
        grad_var = .5 / var * (-N + (sqErr + TrPhiPhiC) / var)

        grad_lambda = .5 / lambda_ * ((TrC + mm) / lambda_ - D)

        grad_thetas = []
        dPhis = basis_func(X, *thetas).grad if len(thetas) > 0 else []
        for dPhi in dPhis:
            dPhiPhi = np.dot(dPhi.T, Phi)
            dt = (np.dot(m.T, np.dot(Err, dPhi)) - np.sum(dPhiPhi * C)) / var
            grad_thetas.append(-dt)

        return FuncRes(value=-elb, grad=(-grad_var, -grad_lambda) + tuple(grad_thetas))

    return elbo


def learn(X, y, basis=identity, basis_args=(), basis_args_bounds=[], var=1., 
          regulariser=1., var_bound=Positive(), regulariser_bound=Positive(),
          tol=1e-6, maxiter=1000, use_autograd=False, minimizer_cb=print,
          verbose=True):

    cache = Bunch()
    elbo = make_elbo(X, y, basis, cache=cache)
    # neg_elbo = func_negate(elbo)

    if use_autograd:
        elbo = value_and_multigrad(func_value(elbo),
                                   argnums=list(range(len(basis_args_bounds) + 2)))

    res = minimize(elbo, method='L-BFGS-B', jac=True,
                   ndarrays=(var, regulariser) + tuple(basis_args),
                   bounds=(var_bound, regulariser_bound) + tuple(basis_args_bounds),
                   callback=minimizer_cb, tol=tol, options=dict(maxiter=maxiter))

    var, regulariser, *basis_args = res.x

    return basis_args, cache.m, cache.C, var


def learn_sgd(X, y, basis, bparams, var=1., regulariser=1., diagcov=False,
              gtol=1e-3, passes=100, rate=0.9, eta=1e-6, batchsize=100,
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
        diagcov: bool, optional
            approximate posterior covariance with diagional matrix (enables
            many features to be used by avoiding a matrix inversion).
        gtol: float,
            SGD tolerance convergence criterion.
        passes: int, optional
            Number of complete passes through the data before optimization
            terminates (unless it converges first).
        rate: float, optional
            SGD decay rate, must be [0, 1].
        eta: float, optional
            Jitter term for adadelta SGD. Ignored if :code:`use_sgd=False`.
        batchsize: int, optional
            number of observations to use per SGD batch.
        verbose: bool, optional
            log the learning status.

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

        Furthermore, since SGD is used to estimate all of the parameters of the
        covariance matrix (or rather a log-cholesky factor), many passes may be
        required for convergence.

        When :code:`diagcov` is :code:`True`, this algorithm still has to
        perform a matix inversion on the posterior weight covariances, and so
        this setting is not efficient when the dimensionality of the features
        is large.
    """

    # Some consistency checking
    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]

    # Initialise parameters
    minit = np.random.randn(D)
    Sinit = gamma.rvs(2, scale=0.5, size=D)
    if not diagcov:
        Sinit = np.diag(np.sqrt(Sinit))[np.tril_indices(D)]

    def ELBO(m, S, _var, _lambda, _theta, Data):

        y, X = Data[:, 0], Data[:, 1:]
        M = len(y)
        B = N / M

        # Get Basis
        Phi = basis(X, *_theta)                      # Nb x D

        # Common computations
        Err = y - Phi.dot(m)
        sqErr = (Err**2).sum()
        mm = (m**2).sum()

        if diagcov:
            C = S
            PPdiag = (Phi**2).sum(axis=0)
            TrPhiPhiC = (PPdiag * C).sum()
            TrC = C.sum()
            logdetC = np.log(C).sum()
        else:
            PhiPhi = Phi.T.dot(Phi)
            LC, C = _logcholfact(S, D)
            TrPhiPhiC = np.sum(PhiPhi * C)
            TrC = np.trace(C)
            logdetC = cho_log_det(LC)

        # Calculate ELBO
        ELBO = -0.5 * (B * (M * np.log(2 * np.pi * _var)
                            + sqErr / _var
                            + TrPhiPhiC / _var)
                       + (TrC + mm) / _lambda
                       - logdetC
                       + D * np.log(_lambda)
                       - D)

        if verbose:
            log.info("ELBO = {}, var = {}, reg = {}, bparams = {}."
                     .format(ELBO, _var, _lambda, _theta))

        # Mean gradient
        dm = B * Err.dot(Phi) / _var - m / _lambda

        # Covariance gradient
        if diagcov:
            dS = - 0.5 * (B * PPdiag / _var + 1. / _lambda - 1. / S)
        else:
            dS = _logcholfact_grad(- (B * PhiPhi.dot(LC) / _var
                                      + LC / _lambda
                                      - cho_solve((LC, True), LC)), LC)

        # Grad variance
        dvar = 0.5 / _var * (-N + B * (TrPhiPhiC + sqErr) / _var)

        # Grad reg
        dlambda = 0.5 / _lambda * ((TrC + mm) / _lambda - D)

        # Loop through basis param grads
        dtheta = []
        dPhis = basis.grad(X, *_theta) if len(_theta) > 0 else []
        for dPhi in dPhis:
            dPhiPhi = (dPhi * Phi).sum(axis=0) if diagcov else dPhi.T.dot(Phi)
            dt = B * (m.T.dot(Err.dot(dPhi)) - (dPhiPhi * C).sum()) / _var
            dtheta.append(-dt)

        return -ELBO, [-dm, -dS, -dvar, -dlambda, dtheta]

    vparams = [minit, Sinit, var, regulariser, bparams]
    bounds = [Bound(shape=minit.shape),
              Positive(shape=Sinit.shape) if diagcov else
              Bound(shape=Sinit.shape),
              Positive(),
              Positive(), basis.bounds]

    nsgd = structured_sgd(logtrick_sgd(sgd))
    res = nsgd(ELBO, vparams, Data=np.hstack((y[:, np.newaxis], X)), rate=rate,
               eta=eta, bounds=bounds, gtol=gtol, passes=passes,
               batchsize=batchsize, eval_obj=True)

    m, S, var, regulariser, bparams = res.x
    C = S if diagcov else _logcholfact(S, D)[1]

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        log.info('Termination condition: {}.'.format(res['message']))

    return m, C, bparams, var


def predict(Xs, basis, basis_args, m, C, var):
    """
    Predict using Bayesian linear regression.

    Parameters
    ----------
        Xs: ndarray
            (Ns,d) array query input dataset (Ns samples, d dimensions).
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of hyperparameters of the basis object.
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
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

    Phi_s = basis(Xs, *basis_args).value

    Ey = Phi_s.dot(m)
    if C.ndim == 2:
        Vf = np.sum(Phi_s.dot(C) * Phi_s, axis=1)
    else:
        Vf = np.sum((Phi_s * C) * Phi_s, axis=1)

    return Ey, Vf, Vf + var


#
# Module Helper functions
#

def _logcholfact(l, D):

    L = np.zeros((D, D))
    L[np.tril_indices(D)] = l
    L[np.diag_indices(D)] = np.exp(L[np.diag_indices(D)])

    return L, L.dot(L.T)


def _logcholfact_grad(dL, L):

    D = dL.shape[0]
    dL[np.diag_indices(D)] *= L[np.diag_indices(D)]
    return dL[np.tril_indices(D)]
