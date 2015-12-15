"""
Implementation of Bayesian GLMs with nonparametric variational inference [1]_,
with a few modifications and tweaks.

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
"""


from __future__ import division

import numpy as np
import logging
from multiprocessing import Pool
from scipy.stats.distributions import gamma
from scipy.optimize import brentq

from .transforms import logsumexp
from .optimize import minimize, sgd
from .utils import CatParameters, Bound, Positive, checktypes, \
    list_to_params as l2p

# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, likelihood, lparams, basis, bparams, reg=1., postcomp=10,
          use_sgd=True, maxit=1000, tol=1e-7, batchsize=100, rate=0.9,
          eta=1e-5, verbose=True):
    """
    Learn the parameters of a Bayesian generalised linear model (GLM).

    The learning algorithm uses nonparametric variational inference [1]_, and
    optionally stochastic gradients.

    Parameters
    ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        likelihood: Object
            A likelihood object, see the likelihoods module.
        lparams: sequence
            a sequence of parameters for the likelihood object, e.g. the
            likelihoods.Gaussian object takes a variance parameter, so this
            should be :code:`[var]`.
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        reg: float, optional
            weight regulariser (variance) initial value.
        postcomp: int, optional
            Number of diagonal Gaussian components to use to approximate the
            posterior distribution.
        tol: float, optional
           Optimiser relative tolerance convergence criterion.
        use_sgd: bool, optional
            If :code:`True` then use SGD (Adadelta) optimisation instead of
            L-BFGS.
        maxit: int, optional
            Maximum number of iterations of the optimiser to run. If
            :code:`use_sgd` is :code:`True` then this is the number of complete
            passes through the data before optimization terminates (unless it
            converges first).
        batchsize: int, optional
            number of observations to use per SGD batch. Ignored if
            :code:`use_sgd=False`.
        rate: float, optional
            SGD decay rate, must be [0, 1]. Ignored if :code:`use_sgd=False`.
        eta: float, optional
            Jitter term for adadelta SGD. Ignored if :code:`use_sgd=False`.
        verbose: bool, optional
            log the learning status.

    Returns
    -------
        m: ndarray
            (D, postcomp) array of posterior weight means (D is the dimension
            of the features).
        C: ndarray
            (D, postcomp) array of posterior weight variances.
        lparams: sequence
            learned sequence of likelihood object hyperparameters.
        bparams: sequence
            learned sequence of basis object hyperparameters.

    Notes
    -----
        This approximates the posterior distribution over the weights with
        a mixture of Gaussians:

        .. math ::

            \mathbf{w} \sim \\frac{1}{K} \sum^K_{k=1}
                \mathcal{N}(\mathbf{m_k}, \\boldsymbol{\Psi}_k)

        where,

        .. math ::

            \\boldsymbol{\Psi}_k = \\text{diag}([\Psi_{k,1}, \ldots,
                \Psi_{k,D}]).

        This is so arbitrary likelihoods can be used with this algorithm, while
        still mainting flexible and tractable non-Gaussian posteriors.
        Additionaly this has the benefit that we have a reduced number of
        parameters to optimise (compared with full covariance Gaussians).

        The main differences between this implementation and the GLM in [1]_
        are:
            - We use diagonal mixtures, as opposed to isotropic.
            - We do not cycle between optimising eq. 10 and 11 (objectives L1
              and L2) in the paper. We use the full objective L2 for
              everything, including the posterior means, and we optimise all
              parameters together.

        Even though these changes make learning a little slower, and require
        third derivatives of the likelihoods, we obtain better results and we
        can use SGD straight-forwardly.
    """

    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *bparams).shape[1]
    K = postcomp

    # Pre-allocate here
    dm = np.zeros((D, K))
    dC = np.zeros((D, K))
    H = np.empty((D, K))

    # Objective function Eq. 10 from [1], and gradients of ALL params
    def L2(params, data):

        # Extract data, parameters, etc
        y, X = data[:, 0], data[:, 1:]
        uparams = pcat.unflatten(params)
        _m, _C, _reg, _lparams, _bparams = uparams

        # Dimensions
        M, d = X.shape
        D, K = _m.shape
        B = N / M

        # Basis function stuff
        Phi = basis(X, *_bparams)  # N x D
        Phi2 = Phi**2
        Phi3 = Phi**3
        dPhi = basis.grad(X, *_bparams)
        dPhiPhi = [dP * Phi for dP in dPhi]
        f = Phi.dot(_m)  # N x K

        # Posterior responsability terms
        logqkk = _qmatrix(_m, _C)
        logqk = logsumexp(logqkk, axis=0)  # log term of Eq. 7 from [1]
        pz = np.exp(logqkk - logqk)

        # Big loop though posterior mixtures for calculating stuff
        ll = 0
        dlp = [np.zeros_like(p) for p in _lparams]
        # dbp = [np.zeros_like(p) for p in _bparams]
        dbp = np.zeros(len(dPhi))

        for k in range(K):

            # Common likelihood calculations
            ll += B * likelihood.loglike(y, f[:, k], *_lparams).sum()
            df = B * likelihood.df(y, f[:, k], *_lparams)
            d2f = B * likelihood.d2f(y, f[:, k], *_lparams)
            d3f = B * likelihood.d3f(y, f[:, k], *_lparams)
            H[:, k] = d2f.dot(Phi2) - 1. / _reg

            # Posterior mean and covariance gradients
            mkmj = _m[:, k][:, np.newaxis] - _m
            iCkCj = 1 / (_C[:, k][:, np.newaxis] + _C)
            dC[:, k] = (-((mkmj * iCkCj)**2 - 2 * iCkCj).dot(pz[:, k])
                        + H[:, k]) / (2 * K)
            dm[:, k] = (df.dot(Phi) + 0.5 * _C[:, k] * d3f.dot(Phi3)
                        + (iCkCj * mkmj).dot(pz[:, k])
                        - _m[:, k] / _reg) / K

            # Likelihood parameter gradients
            dp = likelihood.dp(y, f[:, k], *_lparams)
            dp2df = likelihood.dpd2f(y, f[:, k], *_lparams)
            for l in range(len(_lparams)):
                dpH = dp2df[l].dot(Phi2)
                dlp[l] += B * (dp[l].sum() + 0.5 * (_C[:, k] * dpH).sum()) / K

            # Basis function parameter gradients
            # for l in range(len(_bparams)):
            for l in range(len(dPhi)):
                dPhimk = dPhi[l].dot(_m[:, k])
                dPhiH = d2f.dot(dPhiPhi[l]) + 0.5 * (d3f * dPhimk).dot(Phi2)
                dbp[l] += (df.dot(dPhimk) + (_C[:, k] * dPhiH).sum()) / K

        # Reconstruct dtheta in shape of theta, NOTE: this is a bit clunky!
        dbp = l2p(_bparams, dbp)

        # Regulariser gradient
        dreg = (((_m**2).sum() + _C.sum()) / _reg**2 - D * K / _reg) / (2 * K)

        # Objective, Eq. 10 in [1]
        L2 = 1. / K * (ll
                       - 0.5 * D * K * np.log(2 * np.pi * _reg)
                       - 0.5 * (_m**2).sum() / _reg
                       + 0.5 * (_C * H).sum()
                       - logqk.sum() + np.log(K))

        if verbose:
            log.info("L2 = {}, reg = {}, lparams = {}, bparams = {}"
                     .format(L2, _reg, _lparams, _bparams))

        return -L2, -pcat.flatten_grads(uparams, [dm, dC, dreg, dlp, dbp])

    # Intialise m and C
    m = np.random.randn(D, K) + np.arange(K) - K / 2
    C = gamma.rvs(2, scale=0.5, size=(D, K))

    # Optimiser boiler plate for bounds, log trick, etc
    # NOTE: It would be nice if the optimizer knew how to handle Positive
    # bounds when the log trick is used, so we dont have to have this boiler
    # plate...
    bounds = [Bound()] * (2 * np.prod(m.shape) + 1)
    loginds = [1, 2]
    if checktypes(likelihood.bounds, Positive):
        loginds.append(3)
        bounds += [Bound()] * len(likelihood.bounds)
    else:
        bounds += likelihood.bounds
    if checktypes(basis.bounds, Positive):
        loginds.append(4)
        bounds += [Bound()] * len(basis.bounds)
    else:
        bounds += basis.bounds
    pcat = CatParameters([m, C, reg, lparams, bparams], log_indices=loginds)

    if use_sgd is False:
        res = minimize(L2, pcat.flatten([m, C, reg, lparams, bparams]),
                       ftol=tol, maxiter=maxit, method='L-BFGS-B', jac=True,
                       bounds=bounds, args=(np.hstack((y[:, np.newaxis], X)),))
    else:
        res = sgd(L2, pcat.flatten([m, C, reg, lparams, bparams]),
                  np.hstack((y[:, np.newaxis], X)), rate=rate, eta=eta,
                  bounds=bounds, gtol=tol, passes=maxit, batchsize=batchsize,
                  eval_obj=True)

    m, C, reg, lparams, bparams = pcat.unflatten(res.x)

    if verbose:
        log.info("Finished! Objective = {}, reg = {}, lparams = {}, "
                 "bparams = {}, message: {}."
                 .format(-res.fun, reg, lparams, bparams, res.message))

    return m, C, lparams, bparams


def predict_meanvar(Xs, likelihood, basis, m, C, lparams, bparams,
                    nsamples=100):
    """
    Predictive mean and variance of a Bayesian GLM.

    Parameters
    ----------
        Xs: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        likelihood: Object
            A likelihood object, see the likelihoods module.
        basis: Basis
            A basis object, see the basis_functions module.
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
        lparams: sequence
            a sequence of parameters for the likelihood object, e.g. the
            likelihoods.Gaussian object takes a variance parameter, so this
            should be :code:`[var]`.
        bparams: sequence
            A sequence of hyperparameters of the basis object.
        nsamples: int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.

    Returns
    -------
        Ey: ndarray
            The expected value of ys for the query inputs, Xs of shape (Ns,).
        Vy: ndarray
            The expected variance of ys (excluding likelihood noise terms) for
            the query inputs, Xs of shape (Ns,).
        Ey_min: ndarray
            The minimum sampled values of the predicted mean (same shape as Ey)
        Ey_max: ndarray
            The maximum sampled values of the predicted mean (same shape as Ey)
    """

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    ys = likelihood.Ey(f, *lparams)
    Ey = ys.mean(axis=1)
    Vy = ((ys - Ey[:, np.newaxis])**2).sum(axis=1) / nsamples
    return Ey, Vy, ys.min(axis=1), ys.max(axis=1)


def predict_cdf(quantile, Xs, likelihood, basis, m, C, lparams, bparams,
                nsamples=100):
    """
    Predictive cumulative density function of a Bayesian GLM.

    Parameters
    ----------
        quantile: float
            The predictive probability, :math:`p(y^* \leq \\text{quantile} |
            \mathbf{X}, y)`.
        Xs: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        likelihood: Object
            A likelihood object, see the likelihoods module.
        basis: Basis
            A basis object, see the basis_functions module.
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
        lparams: sequence
            a sequence of parameters for the likelihood object, e.g. the
            likelihoods.Gaussian object takes a variance parameter, so this
            should be :code:`[var]`.
        bparams: sequence
            A sequence of hyperparameters of the basis object.
        nsamples: int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.

    Returns
    -------
        p: ndarray
           The probability of ys <= quantile for the query inputs, Xs of shape
           (Ns,).
        p_min: ndarray
            The minimum sampled values of the predicted probability (same shape
            as p)
        p_max: ndarray
            The maximum sampled values of the predicted probability (same shape
            as p)
    """

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    ps = likelihood.cdf(quantile, f, *lparams)
    return ps.mean(axis=1), ps.min(axis=1), ps.max(axis=1)


def predict_interval(alpha, Xs, likelihood, basis, m, C, lparams, bparams,
                     nsamples=100, multiproc=True):
    """
    Predictive percentile interval (upper and lower quantiles) for a Bayesian
    GLM.

    Parameters
    ----------
        alpha: float
            The percentile confidence interval (e.g. 95%) to return.
        Xs: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        likelihood: Object
            A likelihood object, see the likelihoods module.
        basis: Basis
            A basis object, see the basis_functions module.
        m: ndarray
            (D,) array of regression weights (posterior).
        C: ndarray
            (D,) or (D, D) array of regression weight covariances (posterior).
        lparams: sequence
            a sequence of parameters for the likelihood object, e.g. the
            likelihoods.Gaussian object takes a variance parameter, so this
            should be :code:`[var]`.
        bparams: sequence
            A sequence of hyperparameters of the basis object.
        nsamples: int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.
        multiproc: bool, optional
            Use multiprocessing to paralellise this prediction computation.

    Returns
    -------
        a: ndarray
            The lower end point of the interval with shape (Ns,)
        b: ndarray
            The upper end point of the interval with shape (Ns,)
    """

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    work = ((fn, likelihood, lparams, alpha) for fn in f)

    if multiproc:
        pool = Pool()
        res = pool.starmap(_rootfinding, work)
        pool.close()
        pool.join()
    else:
        res = [_rootfinding(*w) for w in work]
    ql, qu = zip(*res)

    return np.array(ql), np.array(qu)


#
#  Internal Module Utilities
#

def _rootfinding(fn, likelihood, lparams, alpha):

    # CDF minus percentile for quantile root finding
    predCDF = lambda q, fs, percent: \
        (likelihood.cdf(q, fs, *lparams)).mean() - percent

    lpercent = (1 - alpha) / 2
    upercent = 1 - lpercent
    Eyn = likelihood.Ey(fn, *lparams).mean()
    lb, ub = -100 * max(Eyn, 1), 100 * max(Eyn, 1)

    try:
        qln = brentq(predCDF, a=lb, b=ub, args=(fn, lpercent))
    except ValueError:
        qln = np.nan

    try:
        qun = brentq(predCDF, a=lb, b=ub, args=(fn, upercent))
    except ValueError:
        qun = np.nan

    return qln, qun


def _sample_func(Xs, basis, m, C, bparams, nsamples):

    D, K = m.shape
    w = np.zeros((D, nsamples))

    for k in range(K):
        w += m[:, k:k + 1] + np.random.randn(D, nsamples) \
            * np.sqrt(C[:, k:k + 1])

    w /= K
    return basis(Xs, *bparams).dot(w)


def _dgausll(x, mean, dcov):

    D = len(x)
    return - 0.5 * (D * np.log(2 * np.pi) + np.log(dcov).sum()
                    + ((x - mean)**2 / dcov).sum())


def _qmatrix(m, C, k=None):

    K = m.shape[1]
    if k is None:
        logq = [[_dgausll(m[:, i], m[:, j], C[:, i] + C[:, j])
                 for i in range(K)]
                for j in range(K)]
    else:
        logq = [_dgausll(m[:, k], m[:, j], C[:, k] + C[:, j])
                for j in range(K)]

    return np.array(logq)
