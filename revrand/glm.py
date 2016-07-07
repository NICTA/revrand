"""
Implementation of Bayesian GLMs with nonparametric variational inference [1]_,
with a few modifications and tweaks.

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
"""


from __future__ import division

import numpy as np
import logging
from itertools import chain
from multiprocessing import Pool

from scipy.stats.distributions import gamma
from scipy.optimize import brentq, minimize

from .utils import couple, append_or_extend, atleast_list
from .mathfun.special import logsumexp
from .basis_functions import apply_grad
from .optimize import sgd, structured_sgd, structured_minimizer, logtrick_sgd,\
    logtrick_minimizer, AdaDelta
from .btypes import Bound, Positive, Parameter, get_values


# Set up logging
log = logging.getLogger(__name__)


def learn(X, y, likelihood, basis, regulariser=Parameter(1., Positive()),
          likelihood_args=(), postcomp=10, use_sgd=True, maxiter=1000,
          tol=1e-7, batch_size=100, rho=0.9, epsilon=1e-5):
    r"""
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
    basis: Basis
        A basis object, see the basis_functions module.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    likelihood_args: sequence, optional
        sequence of arguments to pass to the likelihood function. These are
        non-learnable parameters. They can be scalars or arrays of length N.
    postcomp: int, optional
        Number of diagonal Gaussian components to use to approximate the
        posterior distribution.
    use_sgd: bool, optional
        If :code:`True` then use SGD (Adadelta) optimisation instead of L-BFGS.
    maxiter: int, optional
        Maximum number of iterations of the optimiser to run. If
        :code:`use_sgd` is :code:`True` then this is the number of mini batches
        to evaluate before termination.
    tol: float, optional
        Optimiser relative tolerance convergence criterion (only if L-BFGS
        is used as the optimiser).
    batch_size: int, optional
        number of observations to use per SGD batch. Ignored if
        :code:`use_sgd=False`.
    rho: float, optional
        SGD decay rate, must be [0, 1]. Ignored if :code:`use_sgd=False`.
    epsilon: float, optional
        Jitter term for adadelta SGD. Ignored if :code:`use_sgd=False`.

    Returns
    -------
    m: ndarray
        (D, postcomp) array of posterior weight means (D is the dimension of
        the features).
    C: ndarray
        (D, postcomp) array of posterior weight variances.
    likelihood_hypers: sequence
        learned sequence of likelihood object hyperparameters.
    basis_hypers: sequence
        learned sequence of basis object hyperparameters.

    Notes
    -----
    This approximates the posterior distribution over the weights with
    a mixture of Gaussians:

    .. math ::

        \mathbf{w} \sim \frac{1}{K} \sum^K_{k=1}
            \mathcal{N}(\mathbf{m_k}, \boldsymbol{\Psi}_k)

    where,

    .. math ::

        \boldsymbol{\Psi}_k = \text{diag}([\Psi_{k,1}, \ldots,
            \Psi_{k,D}]).

    This is so arbitrary likelihoods can be used with this algorithm, while
    still mainting flexible and tractable non-Gaussian posteriors. Additionaly
    this has the benefit that we have a reduced number of parameters to
    optimise (compared with full covariance Gaussians).

    The main differences between this implementation and the GLM in [1]_ are:
        - We use diagonal mixtures, as opposed to isotropic.
        - We do not cycle between optimising eq. 10 and 11 (objectives L1 and
          L2) in the paper. We use the full objective L2 for everything,
          including the posterior means, and we optimise all parameters
          together.

    Even though these changes make learning a little slower, and require third
    derivatives of the likelihoods, we obtain better results and we can use SGD
    straight-forwardly.

    This uses the python logging module for displaying learning status. To view
    these messages have something like,

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

    # Shapes of things
    N, d = X.shape
    D = basis(np.atleast_2d(X[0, :]), *get_values(basis.params)).shape[1]
    K = postcomp

    # Number of hyperparameters
    nlpams = len(atleast_list(get_values(likelihood.params)))
    nbpams = len(atleast_list(get_values(basis.params)))
    tothypers = nlpams + nbpams

    # Pre-allocate here
    dm = np.zeros((D, K))
    dC = np.zeros((D, K))
    H = np.empty((D, K))

    # Make sure we get list output from likelihood parameter gradients
    lgrads = couple(lambda *a: atleast_list(likelihood.dp(*a)),
                    lambda *a: atleast_list(likelihood.dpd2f(*a)))

    # Objective function Eq. 10 from [1], and gradients of ALL params
    def L2(m, C, reg, *args):

        # Unpack data and likelihood arguments if they exist
        y, X = args[tothypers], args[tothypers + 1]
        largs = args[(tothypers + 2):] if len(likelihood_args) > 0 else ()

        # Extract parameters
        lpars, bpars, = args[:nlpams], args[nlpams:tothypers]
        lpars_largs = tuple(chain(lpars, largs))

        # Batchsize and discount factor
        M, _ = X.shape
        B = N / M

        # Basis function stuff
        Phi = basis(X, *bpars)  # M x D
        Phi2 = Phi**2
        Phi3 = Phi**3
        f = Phi.dot(m)  # M x K
        df, d2f, d3f = np.zeros((M, K)), np.zeros((M, K)), np.zeros((M, K))

        # Posterior responsability terms
        logqkk = _qmatrix(m, C)
        logqk = logsumexp(logqkk, axis=0)  # log term of Eq. 7 from [1]
        pz = np.exp(logqkk - logqk)

        # Zero starts for sums over posterior mixtures
        ll = 0
        dlpars = [np.zeros_like(p) for p in lpars]

        # Big loop though posterior mixtures for calculating stuff
        for k in range(K):

            # Common likelihood calculations
            ll += B * likelihood.loglike(y, f[:, k], *lpars_largs).sum()
            df[:, k] = B * likelihood.df(y, f[:, k], *lpars_largs)
            d2f[:, k] = B * likelihood.d2f(y, f[:, k], *lpars_largs)
            d3f[:, k] = B * likelihood.d3f(y, f[:, k], *lpars_largs)
            H[:, k] = d2f[:, k].dot(Phi2) - 1. / reg

            # Posterior mean and covariance gradients
            mkmj = m[:, k][:, np.newaxis] - m
            iCkCj = 1 / (C[:, k][:, np.newaxis] + C)
            dC[:, k] = (-((mkmj * iCkCj)**2 - 2 * iCkCj).dot(pz[:, k])
                        + H[:, k]) / (2 * K)
            dm[:, k] = (df[:, k].dot(Phi)
                        + 0.5 * C[:, k] * d3f[:, k].dot(Phi3)
                        + (iCkCj * mkmj).dot(pz[:, k])
                        - m[:, k] / reg) / K

            # Likelihood parameter gradients
            ziplgrads = zip(*lgrads(y, f[:, k], *lpars_largs))
            for l, (dp, dp2df) in enumerate(ziplgrads):
                dlpars[l] -= B * \
                    (dp.sum() + 0.5 * (C[:, k] * dp2df.dot(Phi2)).sum()) / K

        # Regulariser gradient
        dreg = 0.5 * (((m**2).sum() + C.sum()) / (reg**2 * K) - D / reg)

        # Basis function parameter gradients
        def dtheta(dPhi):
            dt = 0
            dPhiPhi = dPhi * Phi
            for k in range(K):
                dPhimk = dPhi.dot(m[:, k])
                dPhiH = d2f[:, k].dot(dPhiPhi) + \
                    0.5 * (d3f[:, k] * dPhimk).dot(Phi2)
                dt -= (df[:, k].dot(dPhimk) + (C[:, k] * dPhiH).sum()) / K
            return dt

        dbpars = apply_grad(dtheta, basis.grad(X, *bpars))

        # Objective, Eq. 10 in [1]
        L2 = 1. / K * (ll
                       - 0.5 * D * K * np.log(2 * np.pi * reg)
                       - 0.5 * (m**2).sum() / reg
                       + 0.5 * (C * H).sum()
                       - logqk.sum() + np.log(K))

        log.info("L2 = {}, reg = {}, likelihood_hypers = {}, basis_hypers = {}"
                 .format(L2, reg, lpars, bpars))

        return -L2, append_or_extend([-dm, -dC, -dreg], dlpars, dbpars)

    # Intialise m and C
    m = (np.random.randn(D, K) + np.random.randn(K)) * regulariser.value
    C = gamma.rvs(2, scale=0.5, size=(D, K))

    # Pack params
    params = append_or_extend([Parameter(m, Bound()), Parameter(C, Positive()),
                               regulariser], likelihood.params, basis.params)

    # Pack data
    likelihood_args = _reshape_likelihood_args(likelihood_args, N)
    data = (y, X) + likelihood_args

    if use_sgd is False:
        nmin = structured_minimizer(logtrick_minimizer(minimize))
        res = nmin(L2, params, tol=tol, method='L-BFGS-B', jac=True,
                   args=data, options={'maxiter': maxiter, 'maxcor': 100})
    else:
        nsgd = structured_sgd(logtrick_sgd(sgd))
        updater = AdaDelta(rho=rho, epsilon=epsilon)
        res = nsgd(L2, params, data, maxiter=maxiter, updater=updater,
                   batch_size=batch_size, eval_obj=True)

    # Unpack params
    m, C, regulariser = res.x[:3]
    likelihood_hypers = res.x[3:(3 + nlpams)]
    basis_hypers = res.x[(3 + nlpams):]

    log.info("Finished! Objective = {}, reg = {}, likelihood_hypers = {}, "
             "basis_hypers = {}, message: {}."
             .format(-res.fun, regulariser, likelihood_hypers, basis_hypers,
                     res.message))

    return m, C, likelihood_hypers, basis_hypers


def predict_moments(Xs, likelihood, basis, m, C, likelihood_hypers,
                    basis_hypers, likelihood_args=(), nsamples=100):
    r"""
    Predictive moments, in particular mean and variance, of a Bayesian GLM.

    This function uses Monte-Carlo sampling to evaluate the predictive mean and
    variance of a Bayesian GLM. The exact expressions evaluated are,

    .. math ::

        \mathbb{E}[y^*] = \int g(\mathbf{w}^T \boldsymbol\phi^{*})
            p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

        \mathbb{V}[y^*] = \int \left(g(\mathbf{w}^T \boldsymbol\phi^{*})
            - \mathbb{E}[y^*]\right)^2
            p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

    where :math:`g(\cdot)` is the activation (inverse link) link function used
    by the GLM, and :math:`p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi)` is the
    posterior distribution over weights (from :code:`learn`). Here are few
    concrete examples of how we can use these values,

    - Gaussian likelihood: these are just the predicted mean and variance, see
      :code:`revrand.regression.predict`
    - Bernoulli likelihood: The expected value is the probability, :math:`p(y^*
      = 1)`, i.e. the probability of class one. The variance may not be so
      useful.
    - Poisson likelihood: The expected value is similar conceptually to the
      Gaussian case, and is also a *continuous* value. The median (50%
      quantile) from :code:`predict_interval` is a discrete value. Again, the
      variance in this instance may not be so useful.

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
    likelihood_hypers: sequence
        a sequence of parameters for the likelihood object, e.g. the
        likelihoods.Gaussian object takes a variance parameter, so this should
        be :code:`[var]`.
    basis_hypers: sequence
        A sequence of hyperparameters of the basis object.
    likelihood_args: sequence, optional
        sequence of arguments to pass to the likelihood function. These are
        non-learnable parameters. They can be scalars or arrays of length Ns.
    nsamples: int, optional
        The number of samples to draw from the posterior in order to
        approximate the predictive mean and variance.

    Returns
    -------
    Ey: ndarray
        The expected value of ys for the query inputs, Xs of shape (Ns,).
    Vy: ndarray
        The expected variance of ys (excluding likelihood noise terms) for the
        query inputs, Xs of shape (Ns,).
    Ey_min: ndarray
        The minimum sampled values of the predicted mean (same shape as Ey)
    Ey_max: ndarray
        The maximum sampled values of the predicted mean (same shape as Ey)
    """

    # Get latent function samples
    N = Xs.shape[0]
    ys = np.empty((N, nsamples))
    fsamples = sample_func(Xs, basis, m, C, basis_hypers, nsamples)

    # Push samples though likelihood expected value
    for i, f in enumerate(fsamples):
        ys[:, i] = likelihood.Ey(f, *chain(likelihood_hypers, likelihood_args))

    # Average transformed samples (MC integration)
    Ey = ys.mean(axis=1)
    Vy = ((ys - Ey[:, np.newaxis])**2).sum(axis=1) / nsamples
    Ey_min = ys.min(axis=1)
    Ey_max = ys.max(axis=1)

    return Ey, Vy, Ey_min, Ey_max


def predict_logpdf(ys, Xs, likelihood, basis, m, C, likelihood_hypers,
                   basis_hypers, likelihood_args=(), nsamples=100):
    r"""
    Predictive log-probability density function of a Bayesian GLM.

    Parameters
    ----------
    ys: ndarray
        The test observations of shape (Ns,) to evaluate under,
        :math:`\log p(y^* |\mathbf{x}^*, \mathbf{X}, y)`.
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
    likelihood_hypers: sequence
        a sequence of parameters for the likelihood object, e.g. the
        likelihoods.Gaussian object takes a variance parameter, so this should
        be :code:`[var]`.
    basis_hypers: sequence
        A sequence of hyperparameters of the basis object.
    likelihood_args: sequence, optional
        sequence of arguments to pass to the likelihood function. These are
        non-learnable parameters. They can be scalars or arrays of length Ns.
    nsamples: int, optional
        The number of samples to draw from the posterior in order to
        approximate the predictive mean and variance.

    Returns
    -------
    logp: ndarray
       The log probability of ys given Xs of shape (Ns,).
    logp_min: ndarray
        The minimum sampled values of the predicted log probability (same shape
        as p)
    logp_max: ndarray
        The maximum sampled values of the predicted log probability (same shape
        as p)
    """

    # Get latent function samples
    N = Xs.shape[0]
    ps = np.empty((N, nsamples))
    fsamples = sample_func(Xs, basis, m, C, basis_hypers, nsamples)

    # Push samples though likelihood pdf
    for i, f in enumerate(fsamples):
        ps[:, i] = likelihood.loglike(ys, f, *chain(likelihood_hypers,
                                                    likelihood_args))

    # Average transformed samples (MC integration)
    logp = ps.mean(axis=1)
    logp_min = ps.min(axis=1)
    logp_max = ps.max(axis=1)

    return logp, logp_min, logp_max


def predict_cdf(quantile, Xs, likelihood, basis, m, C, likelihood_hypers,
                basis_hypers, likelihood_args=(), nsamples=100):
    r"""
    Predictive cumulative density function of a Bayesian GLM.

    Parameters
    ----------
    quantile: float
        The predictive probability, :math:`p(y^* \leq \text{quantile} |
        \mathbf{x}^*, \mathbf{X}, y)`.
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
    likelihood_hypers: sequence
        a sequence of parameters for the likelihood object, e.g. the
        likelihoods.Gaussian object takes a variance parameter, so this
        should be :code:`[var]`.
    basis_hypers: sequence
        A sequence of hyperparameters of the basis object.
    likelihood_args: sequence, optional
        sequence of arguments to pass to the likelihood function. These are
        non-learnable parameters. They can be scalars or arrays of length Ns.
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

    # Get latent function samples
    N = Xs.shape[0]
    ps = np.empty((N, nsamples))
    fsamples = sample_func(Xs, basis, m, C, basis_hypers, nsamples)

    # Push samples though likelihood cdf
    for i, f in enumerate(fsamples):
        ps[:, i] = likelihood.cdf(quantile, f,
                                  *chain(likelihood_hypers, likelihood_args))

    # Average transformed samples (MC integration)
    p = ps.mean(axis=1)
    p_min = ps.min(axis=1)
    p_max = ps.max(axis=1)

    return p, p_min, p_max


def predict_interval(alpha, Xs, likelihood, basis, m, C, likelihood_hypers,
                     basis_hypers, likelihood_args=(), nsamples=100,
                     multiproc=True):
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
    likelihood_hypers: sequence
        a sequence of parameters for the likelihood object, e.g. the
        likelihoods.Gaussian object takes a variance parameter, so this should
        be :code:`[var]`.
    basis_hypers: sequence
        A sequence of hyperparameters of the basis object.
    likelihood_args: sequence, optional
        sequence of arguments to pass to the likelihood function. These are
        non-learnable parameters. They can be scalars or arrays of length Ns.
    nsamples: int, optional
        The number of samples to draw from the posterior in order to
        approximate the predictive mean and variance.
    multiproc: bool, optional
        Use multiprocessing to paralellise this prediction computation.

    Returns
    -------
    ql: ndarray
        The lower end point of the interval with shape (Ns,)
    qu: ndarray
        The upper end point of the interval with shape (Ns,)
    """

    N = Xs.shape[0]

    # Generate latent function samples per observation (n in N)
    fsamples = sample_func(Xs, basis, m, C, basis_hypers, nsamples, genaxis=0)

    # Make sure likelihood_args is consistent with work
    if len(likelihood_args) > 0:
        likelihood_args = _reshape_likelihood_args(likelihood_args, N)

    # Now create work for distrbuted workers
    work = ((f[0], likelihood, likelihood_hypers, f[1:], alpha)
            for f in zip(fsamples, *likelihood_args))

    # Distribute sampling and rootfinding
    if multiproc:
        pool = Pool()
        res = pool.map(_star_rootfinding, work)
        pool.close()
        pool.join()
    else:
        res = [_rootfinding(*w) for w in work]

    # Get results of work
    ql, qu = zip(*res)
    return np.array(ql), np.array(qu)


def sample_func(Xs, basis, m, C, basis_hypers, nsamples=100, genaxis=1):
    """
    Generate samples from the posterior latent function mixtures of the GLM for
    query inputs, Xs.

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
    basis_hypers: sequence
        A sequence of hyperparameters of the basis object.
    nsamples: int, optional
        The number of samples to draw from the posterior in order to
        approximate the predictive mean and variance.
    genaxis: int
        Axis to return samples from, i.e.
        - :code:`genaxis=1` will give you one sample at a time of f for ALL
            observations (so it will iterate over nsamples).
        - :code:`genaxis=0` will give you all samples of f for ONE
            observation at a time (so it will iterate through Xs, row by row)

    Yields
    ------
    fsamples: ndarray
        of shape (Ns,) if :code:`genaxis=1` with each call being a sample
        from the mixture of latent functions over all Ns. Or of shape
        (nsamples,) if :code:`genaxis=0`, with each call being a all samples
        for an observation, n in Ns.
    """
    D, K = m.shape

    # Generate weight samples from all mixture components
    k = np.random.randint(0, K, size=(nsamples,))
    w = m[:, k] + np.random.randn(D, nsamples) * np.sqrt(C[:, k])
    Phi = basis(Xs, *basis_hypers)  # Keep this here for speed

    # Now generate latent functions samples either colwise or rowwise
    if genaxis == 1:
        fs = (Phi.dot(ws) for ws in w.T)
    elif genaxis == 0:
        fs = (phi_n.dot(w) for phi_n in Phi)
    else:
        raise ValueError("Invalid axis to generate samples from")

    return fs


#
#  Internal Module Utilities
#


def _reshape_likelihood_args(likelihood_args, N):

    reshape_args = []
    for l in likelihood_args:
        if np.isscalar(l):
            l = l * np.ones(N)

        if (np.shape(l)[0] != N) and (len(l) != 0):
            raise ValueError("Likelihood arguments not a compatible shape!")

        reshape_args.append(l)

    return tuple(reshape_args)


# For python 2.7 compatibility instead of pool.starmap
def _star_rootfinding(args):

    return _rootfinding(*args)


def _rootfinding(fn, likelihood, likelihood_hypers, likelihood_args, alpha):

    # CDF minus percentile for quantile root finding
    predCDF = lambda q, fs, percent: \
        (likelihood.cdf(q, fs, *chain(likelihood_hypers,
                                      likelihood_args))).mean() - percent

    # Convert alpha into percentages and get (conservative) bounds for brentq
    lpercent = (1 - alpha) / 2
    upercent = 1 - lpercent
    Eyn = likelihood.Ey(fn, *chain(likelihood_hypers, likelihood_args)).mean()
    lb, ub = -1000 * max(Eyn, 1), 1000 * max(Eyn, 1)

    # Do the root finding optimisation for upper and lower quantiles
    try:
        qln = brentq(predCDF, a=lb, b=ub, args=(fn, lpercent))
    except ValueError:
        qln = np.nan

    try:
        qun = brentq(predCDF, a=lb, b=ub, args=(fn, upercent))
    except ValueError:
        qun = np.nan

    return qln, qun


def _dgausll(x, mean, dcov):
    # This is faster than calling scipy.stats.norm.logpdf

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
