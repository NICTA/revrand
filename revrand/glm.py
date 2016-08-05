"""Bayesian Generalised Linear Model implementation.

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

from .utils import couple, atleast_list
from .mathfun.special import logsumexp
from .basis_functions import apply_grad
from .optimize import sgd, structured_sgd, logtrick_sgd
from .btypes import Bound, Positive, Parameter


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
                 regulariser=Parameter(10., Positive()),
                 postcomp=10,
                 maxiter=3000,
                 batch_size=10,
                 batch_weight=10,
                 updater=None):

        self.like = likelihood
        self.basis = basis
        self.regulariser_init = regulariser
        self.postcomp = postcomp
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.updater = updater

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
        D = self.basis.get_dim(X)

        # Batch magnification factor
        # self.B = K * N / self.batch_size
        self.B = N / self.batch_size

        # Pack data
        likelihood_args = _reshape_likelihood_args(likelihood_args, N)
        data = (X, y) + likelihood_args

        # Intialise weights and covariances
        log.info("Initialising weights by optimising MAP")
        res = sgd(self._map,
                  np.random.randn(D),
                  data,
                  maxiter=self.maxiter,
                  updater=self.updater,
                  batch_size=self.batch_size
                  )

        self.covariance = gamma.rvs(2, scale=0.5, size=(D, K))
        self.weights = self.covariance * np.random.rand(D, K) \
            + res.x[:, np.newaxis]
        self.weights[:, 0] = res.x

        # Pack params
        params = [Parameter(self.weights, Bound()),
                  Parameter(self.covariance, Positive()),
                  self.regulariser_init,
                  self.like.params,
                  self.basis.params]

        log.info("Optimising parameters.")
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
        (self.weights,
         self.covariance,
         self.regulariser,
         self.like_hypers,
         self.basis_hypers
         ) = res.x

        # Store a summary of the optimisation
        self.obj_trace = res.objs
        self.grad_trace = res.norms

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

    def _l2(self, m, C, reg, lpars, bpars, X, y, *largs):
        # Objective function Eq. 10 from [1], and gradients of ALL params

        # Shapes
        D, K = m.shape

        # Make sure hypers and args can be unpacked into callables
        largs = tuple(chain(atleast_list(lpars), largs))

        # Basis function stuff
        Phi = self.basis.transform(X, *atleast_list(bpars))  # M x D
        Phi2 = Phi**2
        Phi3 = Phi**3
        f = Phi.dot(m)  # M x K

        # Posterior responsability terms
        logNkl = _qmatrix(m, C)
        logzk = logsumexp(logNkl, axis=0)  # log term of Eq. 7 from [1]

        # Zero starts for sums over posterior mixtures
        ll = 0
        dlpars = [np.zeros_like(p) for p in atleast_list(lpars)]

        # Big loop though posterior mixtures for calculating stuff
        for k in range(K):

            # Common likelihood calculations
            ll += self.B * self.like.loglike(y, f[:, k], *largs).sum()
            self.df[:, k] = self.B * self.like.df(y, f[:, k], *largs)
            self.d2f[:, k] = self.B * self.like.d2f(y, f[:, k], *largs)
            self.d3f[:, k] = self.B * self.like.d3f(y, f[:, k], *largs)
            self.H[:, k] = self.d2f[:, k].dot(Phi2) - 1. / reg

            # Weight factors for each component in the gradients
            Nkl_zk = np.exp(logNkl[:, k] - logzk[k])
            Nkl_zl = np.exp(logNkl[:, k] - logzk)
            alpha = (Nkl_zk + Nkl_zl)

            # Posterior mean and covariance gradients
            mkmj = m[:, k][:, np.newaxis] - m
            iCkCj = 1 / (C[:, k][:, np.newaxis] + C)
            self.dC[:, k] = ((iCkCj - (mkmj * iCkCj)**2).dot(alpha / K)
                             + self.H[:, k]) / (2 * K)
            self.dm[:, k] = (self.df[:, k].dot(Phi)
                             + 0.5 * C[:, k] * self.d3f[:, k].dot(Phi3)
                             + (iCkCj * mkmj).dot(alpha / K)
                             - m[:, k] / reg) / K

            # Likelihood parameter gradients
            dp = atleast_list(self.like.dp(y, f[:, k], *largs))
            dpd2f = atleast_list(self.like.dpd2f(y, f[:, k], *largs))
            for l, (dp_l, dp2df_l) in enumerate(zip(dp, dpd2f)):
                dlpars[l] -= self.B / K \
                    * (dp_l.sum() + 0.5 * (C[:, k] * dp2df_l.dot(Phi2)).sum())

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

        dbpars = apply_grad(dtheta, self.basis.grad(X, *atleast_list(bpars)))

        # Objective, Eq. 10 in [1]
        L2 = (ll
              - 0.5 * D * K * np.log(2 * np.pi * reg)
              - 0.5 * (m**2).sum() / reg
              + 0.5 * (C * self.H).sum()
              - (logzk / K).sum()) / K

        log.info("L2 = {}, reg = {}, likelihood_hypers = {}, basis_hypers = {}"
                 .format(L2, reg, lpars, bpars))

        return -L2, [-self.dm, -self.dC, -dreg, dlpars, dbpars]

    def _map(self, weights, X, y, *largs):
        # MAP objective for initialising the weights

        # Extract parameters from their initial values
        bpars = atleast_list(self.basis.params.value)
        largs = tuple(chain(atleast_list(self.like.params.value), largs))
        reg = self.regulariser_init.value

        # Gradient
        Phi = self.basis.transform(X, *bpars)
        f = Phi.dot(weights)
        df = self.like.df(y, f, *largs)
        dweights = self.B * df.dot(Phi) - weights / reg

        return -dweights

    def predict(self, X, likelihood_args=(), nsamples=100):
        """
        Predict target values from Bayesian generalised linear regression.

        Parameters
        ----------
        X: ndarray
            (Ns,d) array query input dataset (Ns samples, d dimensions).
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
            The expected value of y_star for the query inputs, X_star of shape
            (N_star,).
        """
        Ey, _, _, _ = self.predict_moments(X, likelihood_args, nsamples)

        return Ey

    def predict_moments(self, X, likelihood_args=(), nsamples=100):
        r"""
        Predictive moments, in particular mean and variance, of a Bayesian GLM.

        This function uses Monte-Carlo sampling to evaluate the predictive mean
        and variance of a Bayesian GLM. The exact expressions evaluated are,

        .. math ::

            \mathbb{E}[y^* | \mathbf{x^*}, \mathbf{X}, y] &=
                \int \mathbb{E}[y^* | \mathbf{w}, \phi(\mathbf{x}^*)]
                p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

            \mathbb{V}[y^* | \mathbf{x^*}, \mathbf{X}, y] &=
                \int \left(\mathbb{E}[y^* | \mathbf{w}, \phi(\mathbf{x}^*)]
                - \mathbb{E}[y^* | \mathbf{x^*}, \mathbf{X}, y]\right)^2
                p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi) d\mathbf{w},

        where :math:`\mathbb{E}[y^* | \mathbf{w}, \phi(\mathbf{x}^*)]` is the
        the expected value of :math:`y^*` from  the likelihood, and
        :math:`p(\mathbf{w} | \mathbf{y}, \boldsymbol\Phi)` is the posterior
        distribution over weights (from :code:`learn`). Here are few concrete
        examples of how we can use these values,

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
        Ey_args = tuple(chain(atleast_list(self.like_hypers), likelihood_args))
        for i, f in enumerate(fsamples):
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
        ll_args = tuple(chain(atleast_list(self.like_hypers), likelihood_args))
        for i, f in enumerate(fsamples):
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
        cdf_arg = tuple(chain(atleast_list(self.like_hypers), likelihood_args))
        for i, f in enumerate(fsamples):
            ps[:, i] = self.like.cdf(quantile, f, *cdf_arg)

        # Average transformed samples (MC integration)
        p = ps.mean(axis=1)
        p_min = ps.min(axis=1)
        p_max = ps.max(axis=1)

        return p, p_min, p_max

    def predict_interval(self, percentile, X, likelihood_args=(), nsamples=100,
                         multiproc=True):
        """
        Predictive percentile interval (upper and lower quantiles).

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
        like_hypers = atleast_list(self.like_hypers)
        work = ((f[0], self.like, like_hypers, f[1:], percentile)
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

        # Do this here for *massive* speed improvements
        Phi = self.basis.transform(X, *atleast_list(self.basis_hypers))

        # Now generate latent functions samples either colwise or rowwise
        if genaxis == 1:
            fs = (Phi.dot(ws) for ws in w.T)
        elif genaxis == 0:
            fs = (phi_n.dot(w) for phi_n in Phi)
        else:
            raise ValueError("Invalid axis to generate samples from")

        return fs


# For US spelling
class GeneralizedLinearModel(GeneralisedLinearModel):

    pass


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
