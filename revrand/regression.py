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

from scipy.linalg import cho_solve
from scipy.stats.distributions import gamma

from .utils import append_or_extend
from .linalg import jitchol, cho_log_det
from .optimize import minimize, sgd, Bound, Positive, structured_minimizer, \
    logtrick_minimizer, structured_sgd, logtrick_sgd
from .basis_functions import apply_grad

# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, basis, bparams, var=1., regulariser=1., diagcov=False,
          ftol=1e-6, maxit=1000, verbose=True):
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

        # Posterior Parameters
        lower = False
        LiC = jitchol(np.diag(np.ones(D) / _lambda) + PhiPhi / _var,
                      lower=lower)
        m = cho_solve((LiC, lower), Phi.T.dot(y)) / _var

        # Common calcs dependent on form of C
        if diagcov:
            C = 1. / (PhiPhi.diagonal() / _var + 1. / _lambda)
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

        # Grad var
        dvar = 0.5 / _var * (-N + (sqErr + TrPhiPhiC) / _var)

        # Grad reg
        dlambda = 0.5 / _lambda * ((TrC + mm) / _lambda - D)

        # Get structured basis function gradients
        def dtheta(dPhi):
            dPhiPhi = (dPhi * Phi).sum(axis=0) if diagcov else dPhi.T.dot(Phi)
            return - (m.T.dot(Err.dot(dPhi)) - (dPhiPhi * C).sum()) / _var

        dtheta = apply_grad(dtheta, basis.grad(X, *_theta))

        # if len(_theta) > 0:
        #     import IPython; IPython.embed()

        return -ELBO, append_or_extend([-dvar, -dlambda], dtheta)

    bounds = append_or_extend([Positive(), Positive()], basis.bounds)
    nmin = structured_minimizer(logtrick_minimizer(minimize))
    res = nmin(ELBO, [var, regulariser] + bparams, method='L-BFGS-B', jac=True,
               bounds=bounds, ftol=ftol, maxiter=maxit)
    (var, regulariser), bparams = res.x[:2], res.x[2:]

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}, "
                 "message = {}."
                 .format(-res['fun'], var, regulariser, bparams, res.message))

    return mcache, Ccache, bparams, var


def learn_sgd(X, y, basis, bparams, var=1, regulariser=1., diagcov=False,
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

    def ELBO(m, S, _var, _lambda, *args):

        _theta, y, X = args[:-1], args[-1][:, 0], args[-1][:, 1:]
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

        # Get structured basis function gradients
        def dtheta(dPhi):
            dPhiPhi = (dPhi * Phi).sum(axis=0) if diagcov else dPhi.T.dot(Phi)
            return - (m.T.dot(Err.dot(dPhi)) - (dPhiPhi * C).sum()) / _var

        dtheta = apply_grad(dtheta, basis.grad(X, *_theta))

        return -ELBO, append_or_extend([-dm, -dS, -dvar, -dlambda], dtheta)

    vparams = [minit, Sinit, var, regulariser] + bparams
    bounds = [Bound(shape=minit.shape),
              Positive(shape=Sinit.shape) if diagcov else
              Bound(shape=Sinit.shape),
              Positive(),
              Positive()]
    append_or_extend(bounds, basis.bounds)

    nsgd = structured_sgd(logtrick_sgd(sgd))
    res = nsgd(ELBO, vparams, Data=np.hstack((y[:, np.newaxis], X)), rate=rate,
               eta=eta, bounds=bounds, gtol=gtol, passes=passes,
               batchsize=batchsize, eval_obj=True)

    (m, S, var, regulariser), bparams = res.x[:4], res.x[4:]
    C = S if diagcov else _logcholfact(S, D)[1]

    if verbose:
        log.info("Done! ELBO = {}, var = {}, reg = {}, bparams = {}."
                 .format(-res['fun'], var, regulariser, bparams))
        log.info('Termination condition: {}.'.format(res['message']))

    return m, C, bparams, var


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
