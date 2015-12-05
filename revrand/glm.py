"""
Implementation of Bayesian GLMs with nonparametric variational inference [1_],
with a few modifications and tweaks.

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
"""


from __future__ import division

import numpy as np
import logging
from scipy.stats.distributions import gamma

from .transforms import logsumexp
from .optimize import minimize, sgd
from .utils import CatParameters, Bound, Positive, checktypes, \
    list_to_params as l2p

# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, likelihood, lparams, basis, bparams, reg=1., postcomp=10,
          use_sgd=False, maxit=1000, tol=1e-7, batchsize=100, rate=0.9,
          eta=1e-5, verbose=True):
    """
    Learn the parameters and hyperparameters of an Bayesian generalised linear
    model (GLM) using nonparametric variational inference, and optionally
    stochastic gradients.

    Parameters
    ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        likelihood: 
        basis: Basis
            A basis object, see the basis_functions module.
        bparams: sequence
            A sequence of parameters of the basis object.
        var: float, optional
            observation variance initial value.
        regulariser: float, optional
            weight regulariser (variance) initial value.
        rank: int, optional
            the rank of the off-diagonal covariance approximation, has to be
            [0, D] where D is the dimension of the basis. None is the same as
            setting rank = D.
        gtol: float,
            SGD tolerance convergence criterion.
        passes: int, optional
            Number of complete passes through the data before optimization
            terminates (unless it converges first).
        rate: float, optional
            SGD decay rate, must be [0, 1].
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
        This approximates the posterior covariance matrix over the weights with
        a diagonal plus low rank matrix:

        .. math ::

            \mathbf{w} \sim \mathcal{N}(\mathbf{m}, \mathbf{C})

        where,

        .. math ::

            \mathbf{C} = \mathbf{U}\mathbf{U}^{T} + \\text{diag}(\mathbf{s}),
            \quad \mathbf{U} \in \mathbb{R}^{D\\times \\text{rank}},
            \quad \mathbf{s} \in \mathbb{R}^{D}.

        This is to allow for a reduced number of parameters to optimise with
        SGD. As a consequence, features with large dimensionality can be used.
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
        dPhi = basis.grad(X, *_bparams)
        dPhiPhi = [dP * Phi for dP in dPhi]
        PPP = [np.outer(p, p).dot(p) for p in Phi]
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
            iCkCj = 2 / (_C[:, k][:, np.newaxis] + _C)  # TODO: this correct?
            dC[:, k] = (-((mkmj * iCkCj)**2 - iCkCj).dot(pz[:, k])
                        + H[:, k]) / (2 * K)
            dm[:, k] = (df.dot(Phi) + 0.5 * _C[:, k] * d3f.dot(PPP)
                        - (pz[:, k] * iCkCj * mkmj).sum(axis=1)
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
    # m = np.random.randn(D, K) + (np.arange(K) - K / 2)
    m = np.random.randn(D, K)
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

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    ys = likelihood.Ey(f, *lparams)
    Ey = ys.mean(axis=1)
    Vy = ((ys - Ey[:, np.newaxis])**2).sum(axis=1) / nsamples
    return Ey, Vy, ys.min(axis=1), ys.max(axis=1)


def predict_cdf(quantile, Xs, likelihood, basis, m, C, lparams, bparams,
                nsamples=100):

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    ps = likelihood.cdf(quantile, f, *lparams)
    return ps.mean(axis=1), ps.min(axis=1), ps.max(axis=1)


def predict_interval(alpha, Xs, likelihood, basis, m, C, lparams, bparams,
                     nsamples=100):

    f = _sample_func(Xs, basis, m, C, bparams, nsamples)
    a, b = likelihood.interval(alpha, f, *lparams)
    return a.mean(axis=1), b.mean(axis=1)


#
# Private module functions
#

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
