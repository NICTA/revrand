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
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .utils import couple, append_or_extend, atleast_list
from .mathfun.special import logsumexp
from .basis_functions import apply_grad
from .optimize import sgd, structured_sgd, logtrick_sgd
from .btypes import Bound, Positive, Parameter, get_values


# Set up logging
log = logging.getLogger(__name__)


class GeneralisedLinearModel(BaseEstimator, RegressorMixin):
    r"""
    Bayesian Generalised linear model (GLM).

    This provides a scikit learn compatible interface for the glm module.

    Parameters
    ----------
    likelihood: Object
        A likelihood object, see the likelihoods module.
    basis: Basis
        A basis object, see the basis_functions module.
    regulariser: Parameter, optional
        weight regulariser (variance) initial value.
    postcomp: int, optional
        Number of diagonal Gaussian components to use to approximate the
        posterior distribution.
    tol: float, optional
       Optimiser relative tolerance convergence criterion.
    maxiter: int, optional
        Maximum number of iterations of stochastic gradients to run.
    batch_size: int, optional
        number of observations to use per SGD batch.
    updater: SGDUpdater, optional
        The SGD learning rate updating algorithm to use, by default this is
        AdaDelta. See revrand.optimize.sgd for different options.

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

    def __init__(self,
                 likelihood,
                 basis,
                 regulariser=Parameter(1., Positive()),
                 postcomp=10,
                 maxiter=3000,
                 tol=1e-8,
                 batch_size=10,
                 updater=None):

        self.like = likelihood
        self.basis = basis
        self.regulariser_init = regulariser
        self.postcomp = postcomp
        self.maxiter = maxiter
        self.tol = tol
        self.batch_size = batch_size
        self.updater = updater

        # Number of hyperparameters
        self._nlpams = len(atleast_list(get_values(self.like.params)))
        self._nbpams = len(atleast_list(get_values(self.basis.params)))
        self._tothypers = self._nlpams + self._nbpams

        # Make sure we get list output from likelihood parameter gradients
        self.lgrads = couple(lambda *a: atleast_list(self.like.dp(*a)),
                             lambda *a: atleast_list(self.like.dpd2f(*a)))

    def fit(self, X, y, likelihood_args=()):
        r"""
        Learn the parameters of a Bayesian generalised linear model (GLM).

        The learning algorithm uses nonparametric variational inference [1]_,
        and optionally stochastic gradients.

        Parameters
        ----------
        X: ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y: ndarray
            (N,) array targets (N samples)
        likelihood: Object
            A likelihood object, see the likelihoods module.
        likelihood_args: sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N.
        """

        X, y = check_X_y(X, y)

        # Shapes of things
        K = self.postcomp
        N, _ = X.shape
        D = self.basis.transform(np.atleast_2d(X[0, :]),
                                 *get_values(self.basis.params)).shape[1]

        # Batchsize and discount factor
        self.B = N / self.batch_size

        # Intialise weights and covariances
        self.weights = (np.random.randn(D, K) + np.random.randn(K)) \
            * self.regulariser_init.value
        self.covariance = gamma.rvs(2, scale=0.5, size=(D, K))

        # Pack params
        params = append_or_extend([Parameter(self.weights, Bound()),
                                   Parameter(self.covariance, Positive()),
                                   self.regulariser_init],
                                  self.like.params,
                                  self.basis.params)

        # Pack data
        likelihood_args = _reshape_likelihood_args(likelihood_args, N)
        data = (y, X) + likelihood_args

        nsgd = structured_sgd(logtrick_sgd(sgd))

        self._create_cache()
        res = nsgd(self._l2,
                   params,
                   data,
                   maxiter=self.maxiter,
                   updater=self.updater,
                   batch_size=self.batch_size,
                   eval_obj=True
                   )
        self._delete_cache()

        # Unpack params
        self.weights, self.covariance, self.regulariser = res.x[:3]
        self.like_hypers = res.x[3:(3 + self._nlpams)]
        self.basis_hypers = res.x[(3 + self._nlpams):]

        log.info("Finished! Objective = {}, reg = {}, likelihood_hypers = {}, "
                 "basis_hypers = {}, message: {}."
                 .format(-res.fun,
                         self.regulariser,
                         self.like_hypers,
                         self.basis_hypers,
                         res.message))

        return self

    def _create_cache(self):
        # Caching arrays for faster computations

        D, K = self.weights.shape

        self.df = np.zeros((self.batch_size, K))
        self.d2f = np.zeros((self.batch_size, K))
        self.d3f = np.zeros((self.batch_size, K))

        self.dm = np.zeros((D, K))
        self.dC = np.zeros((D, K))
        self.H = np.empty((D, K))

    def _delete_cache(self):
        # Delete the cached arrays

        del self.df, self.d2f, self.d3f, self.dm, self.dC, self.H

    def _l2(self, m, C, reg, *args):
        # Objective function Eq. 10 from [1], and gradients of ALL params

        # Shapes
        D, K = m.shape

        # Unpack data and likelihood arguments if they exist
        y, X = args[self._tothypers], args[self._tothypers + 1]
        largs = ()
        if len(args) > (self._tothypers + 2):
            largs = args[(self._tothypers + 2):]

        # Extract parameters
        lpars, bpars, = args[:self._nlpams], args[self._nlpams:self._tothypers]
        lpars_largs = tuple(chain(lpars, largs))

        # Basis function stuff
        Phi = self.basis.transform(X, *bpars)  # M x D
        Phi2 = Phi**2
        Phi3 = Phi**3
        f = Phi.dot(m)  # M x K

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
            ll += self.B * self.like.loglike(y, f[:, k], *lpars_largs).sum()
            self.df[:, k] = self.B * self.like.df(y, f[:, k], *lpars_largs)
            self.d2f[:, k] = self.B * self.like.d2f(y, f[:, k], *lpars_largs)
            self.d3f[:, k] = self.B * self.like.d3f(y, f[:, k], *lpars_largs)
            self.H[:, k] = self.d2f[:, k].dot(Phi2) - 1. / reg

            # Posterior mean and covariance gradients
            mkmj = m[:, k][:, np.newaxis] - m
            iCkCj = 1 / (C[:, k][:, np.newaxis] + C)
            self.dC[:, k] = (-((mkmj * iCkCj)**2 - 2 * iCkCj).dot(pz[:, k])
                             + self.H[:, k]) / (2 * K)
            self.dm[:, k] = (self.df[:, k].dot(Phi)
                             + 0.5 * C[:, k] * self.d3f[:, k].dot(Phi3)
                             + (iCkCj * mkmj).dot(pz[:, k])
                             - m[:, k] / reg) / K

            # Likelihood parameter gradients
            ziplgrads = zip(*self.lgrads(y, f[:, k], *lpars_largs))
            for l, (dp, dp2df) in enumerate(ziplgrads):
                dlpars[l] -= self.B / K \
                    * (dp.sum() + 0.5 * (C[:, k] * dp2df.dot(Phi2)).sum())

        # Regulariser gradient
        dreg = 0.5 * (((m**2).sum() + C.sum()) / (reg**2 * K) - D / reg)

        # Basis function parameter gradients
        def dtheta(dPhi):
            dt = 0
            dPhiPhi = dPhi * Phi
            for k in range(K):
                dPhimk = dPhi.dot(m[:, k])
                dPhiH = self.d2f[:, k].dot(dPhiPhi) + \
                    0.5 * (self.d3f[:, k] * dPhimk).dot(Phi2)
                dt -= (self.df[:, k].dot(dPhimk) + (C[:, k] * dPhiH).sum()) / K
            return dt

        dbpars = apply_grad(dtheta, self.basis.grad(X, *bpars))

        # Objective, Eq. 10 in [1]
        L2 = (ll
              - 0.5 * D * K * np.log(2 * np.pi * reg)
              - 0.5 * (m**2).sum() / reg
              + 0.5 * (C * self.H).sum()
              - logqk.sum() + np.log(K)) / K

        log.info("L2 = {}, reg = {}, likelihood_hypers = {}, basis_hypers = {}"
                 .format(L2, reg, lpars, bpars))

        return -L2, append_or_extend([-self.dm, -self.dC, -dreg], dlpars,
                                     dbpars)

    def predict(self, X, likelihood_args=(), nsamples=100):

        Ey, _, _, _ = self.predict_moments(X, likelihood_args, nsamples)

        return Ey

    def predict_moments(self, X, likelihood_args=(), nsamples=100):
        r"""
        Predictive moments, in particular mean and variance, of a Bayesian GLM.

        This function uses Monte-Carlo sampling to evaluate the predictive mean
        and variance of a Bayesian GLM. The exact expressions evaluated are,

        .. math ::

            \mathbb{E}[y^*] = \int g(\mathbf{w}^T \boldsymbol\phi^{*})
                p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

            \mathbb{V}[y^*] = \int \left(g(\mathbf{w}^T \boldsymbol\phi^{*})
                - \mathbb{E}[y^*]\right)^2
                p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

        where :math:`g(\cdot)` is the activation (inverse link) link function
        used by the GLM, and :math:`p(\mathbf{w} | \mathbf{y},
        \boldsymbol\Phi)` is the posterior distribution over weights (from
        :code:`learn`). Here are few concrete examples of how we can use these
        values,

        - Gaussian likelihood: these are just the predicted mean and variance,
          see :code:`revrand.regression.predict`
        - Bernoulli likelihood: The expected value is the probability,
          :math:`p(y^* = 1)`, i.e. the probability of class one. The variance
          may not be so useful.
        - Poisson likelihood: The expected value is similar conceptually to the
          Gaussian case, and is also a *continuous* value. The median (50%
          quantile) from :code:`predict_interval` is a discrete value. Again,
          the variance in this instance may not be so useful.

        Parameters
        ----------
        X: ndarray
            (N,d) array query input dataset (Ns samples, D dimensions).
        likelihood_args: sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N.
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

        # Get latent function samples
        N = X.shape[0]
        ys = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood expected value
        for i, f in enumerate(fsamples):
            Ey_args = chain(self.like_hypers, likelihood_args)
            ys[:, i] = self.like.Ey(f, *Ey_args)

        # Average transformed samples (MC integration)
        Ey = ys.mean(axis=1)
        Vy = ((ys - Ey[:, np.newaxis])**2).sum(axis=1) / nsamples
        Ey_min = ys.min(axis=1)
        Ey_max = ys.max(axis=1)

        return Ey, Vy, Ey_min, Ey_max

    def predict_logpdf(self, X, y, likelihood_args=(), nsamples=100):
        r"""
        Predictive log-probability density function of a Bayesian GLM.

        Parameters
        ----------
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        y: float or ndarray
            The test observations of shape (Ns,) to evaluate under,
            :math:`\log p(y^* |\mathbf{x}^*, \mathbf{X}, y)`.
        likelihood_args: sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            Ns.
        nsamples: int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.

        Returns
        -------
        logp: ndarray
           The log probability of ys given Xs of shape (Ns,).
        logp_min: ndarray
            The minimum sampled values of the predicted log probability (same
            shape as p)
        logp_max: ndarray
            The maximum sampled values of the predicted log probability (same
            shape as p)
        """

        X, y = check_X_y(X, y)

        # Get latent function samples
        N = X.shape[0]
        ps = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood pdf
        for i, f in enumerate(fsamples):
            ll_args = chain(self.like_hypers, likelihood_args)
            ps[:, i] = self.like.loglike(y, f, *ll_args)

        # Average transformed samples (MC integration)
        logp = ps.mean(axis=1)
        logp_min = ps.min(axis=1)
        logp_max = ps.max(axis=1)

        return logp, logp_min, logp_max

    def predict_cdf(self, quantile, X, likelihood_args=(), nsamples=100):
        r"""
        Predictive cumulative density function of a Bayesian GLM.

        Parameters
        ----------
        quantile: float
            The predictive probability, :math:`p(y^* \leq \text{quantile} |
            \mathbf{x}^*, \mathbf{X}, y)`.
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        likelihood_args: sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            Ns.
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
        N = X.shape[0]
        ps = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood cdf
        for i, f in enumerate(fsamples):
            cdf_args = chain(self.like_hypers, likelihood_args)
            ps[:, i] = self.like.cdf(quantile, f, *cdf_args)

        # Average transformed samples (MC integration)
        p = ps.mean(axis=1)
        p_min = ps.min(axis=1)
        p_max = ps.max(axis=1)

        return p, p_min, p_max

    def predict_interval(self, percentile, X, likelihood_args=(), nsamples=100,
                         multiproc=True):
        """
        Predictive percentile interval (upper and lower quantiles) for a
        Bayesian GLM.

        Parameters
        ----------
        percentile: float
            The percentile confidence interval (e.g. 95%) to return.
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, D dimensions).
        likelihood_args: sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            Ns.
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

        N = X.shape[0]

        # Generate latent function samples per observation (n in N)
        fsamples = self._sample_func(X, nsamples, genaxis=0)

        # Make sure likelihood_args is consistent with work
        if len(likelihood_args) > 0:
            likelihood_args = _reshape_likelihood_args(likelihood_args, N)

        # Now create work for distrbuted workers
        work = ((f[0], self.like, self.like_hypers, f[1:], percentile)
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

    def _sample_func(self, X, nsamples=100, genaxis=1):
        """
        Generate samples from the posterior latent function mixtures of the GLM
        for query inputs, Xs.

        Parameters
        ----------
        X: ndarray
            (Ns, d) array query input dataset (Ns samples, D dimensions).
        nsamples: int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.
        genaxis: int
            Axis to return samples from, i.e.
            - :code:`genaxis=1` will give you one sample at a time of f for ALL
                observations (so it will iterate over nsamples).
            - :code:`genaxis=0` will give you all samples of f for ONE
                observation at a time (so it will iterate through Xs, row by
                row)

        Yields
        ------
        fsamples: ndarray
            of shape (Ns,) if :code:`genaxis=1` with each call being a sample
            from the mixture of latent functions over all Ns. Or of shape
            (nsamples,) if :code:`genaxis=0`, with each call being a all
            samples for an observation, n in Ns.
        """

        check_is_fitted(self, ['weights', 'covariance', 'basis_hypers',
                               'like_hypers', 'regulariser'])
        X = check_array(X)
        D, K = self.weights.shape

        # Generate weight samples from all mixture components
        k = np.random.randint(0, K, size=(nsamples,))
        w = self.weights[:, k] + np.random.randn(D, nsamples) \
            * np.sqrt(self.covariance[:, k])
        Phi = self.basis.transform(X, *self.basis_hypers)  # Keep this 4 speed

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


def _rootfinding(fn, likelihood, likelihood_hypers, likelihood_args,
                 percentile):

    # CDF minus percentile for quantile root finding
    predCDF = lambda q, fs, percent: \
        (likelihood.cdf(q, fs, *chain(likelihood_hypers,
                                      likelihood_args))).mean() - percent

    # Convert alpha into percentages and get (conservative) bounds for brentq
    lpercent = (1 - percentile) / 2
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
