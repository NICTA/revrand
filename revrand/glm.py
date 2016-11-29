"""Bayesian Generalized Linear Model implementation.

Implementation of Bayesian GLMs using a mixture of Gaussians posterior
approximation and auto-encoding variational Bayes inference. See [1]_ for the
posterior mixture idea, and [2]_ for the inference scheme.

.. [1] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
   inference". Proceedings of the international conference on machine learning.
   2012.
.. [2] Kingma, D. P., & Welling, M. "Auto-encoding variational Bayes".
   Proceedings of the 2nd International Conference on Learning Representations
   (ICLR). 2014.
"""
from __future__ import division

import numpy as np
import logging
from itertools import chain
from multiprocessing import Pool

from scipy.stats.distributions import gamma, norm
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils import check_random_state

from .utils import atleast_list, issequence
from .mathfun.special import logsumexp
from .basis_functions import apply_grad
from .optimize import sgd, structured_sgd, logtrick_sgd
from .btypes import Bound, Positive, Parameter


# Set up logging
log = logging.getLogger(__name__)


# Module settings
WGTRND = norm()  # Sampling distribution over mixture weights
COVRND = gamma(a=2, scale=0.5)  # Sampling distribution over mixture covariance
LOGITER = 500  # Number of SGD iterations between logging ELBO and hypers


class GeneralizedLinearModel(BaseEstimator, RegressorMixin):
    r"""
    Bayesian Generalized linear model (GLM).

    This provides a scikit learn compatible interface for the glm module.

    Parameters
    ----------
    likelihood : Object
        A likelihood object, see the likelihoods module.
    basis : Basis
        A basis object, see the basis_functions module.
    K : int, optional
        Number of diagonal Gaussian components to use to approximate the
        posterior distribution.
    maxiter : int, optional
        Maximum number of iterations of stochastic gradients to run.
    batch_size : int, optional
        number of observations to use per SGD batch.
    updater : SGDUpdater, optional
        The SGD learning rate updating algorithm to use, by default this is
        Adam. See revrand.optimize.sgd for different options.
    nsamples : int, optional
        Number of samples for sampling the expected likelihood and expected
        likelihood gradients
    nstarts : int, optional
        if there are any parameters with distributions as initial values, this
        determines how many random candidate starts shoulds be evaluated before
        commencing optimisation at the best candidate.
    random_state : None, int or RandomState, optional
        random seed

    Notes
    -----
    This approximates the posterior distribution over the weights with a
    mixture of Gaussians:

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
        - We use auto encoding variational Bayes (AEVB) inference [2]_ with
          stochastic gradients.

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
                 K=10,
                 maxiter=3000,
                 batch_size=10,
                 updater=None,
                 nsamples=50,
                 nstarts=500,
                 random_state=None
                 ):
        """See class docstring."""
        self.likelihood = likelihood
        self.basis = basis
        self.K = K
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.updater = updater
        self.nsamples = nsamples
        self.nstarts = nstarts
        self.random_state = random_state  # For clone compatibility
        self.random_ = check_random_state(self.random_state)

    def fit(self, X, y, likelihood_args=()):
        r"""
        Learn the parameters of a Bayesian generalized linear model (GLM).

        Parameters
        ----------
        X : ndarray
            (N, d) array input dataset (N samples, d dimensions).
        y : ndarray
            (N,) array targets (N samples)
        likelihood : Object
            A likelihood object, see the likelihoods module.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N.
        """
        X, y = check_X_y(X, y)

        # Batch magnification factor
        N, _ = X.shape
        self.B_ = X.shape[0] / self.batch_size
        self.D_ = self.basis.get_dim(X)

        # Pack data
        likelihood_args = _reshape_likelihood_args(likelihood_args, N)
        data = (X, y) + likelihood_args

        # Pack params
        params = [Parameter(WGTRND, Bound(), shape=(self.D_, self.K)),
                  Parameter(COVRND, Positive(), shape=(self.D_, self.K)),
                  self.basis.regularizer,
                  self.likelihood.params,
                  self.basis.params]

        log.info("Optimising parameters...")
        self.__it = -self.nstarts  # Keeping track of iterations for logging
        nsgd = structured_sgd(logtrick_sgd(sgd))
        res = nsgd(self._elbo,
                   params,
                   data,
                   eval_obj=True,
                   maxiter=self.maxiter,
                   updater=self.updater,
                   batch_size=self.batch_size,
                   random_state=self.random_,
                   nstarts=self.nstarts
                   )

        # Unpack params
        (self.weights_,
         self.covariance_,
         self.regularizer_,
         self.like_hypers_,
         self.basis_hypers_
         ) = res.x

        log.info("Finished! reg = {}, likelihood_hypers = {}, "
                 "basis_hypers = {}, message: {}."
                 .format(self.regularizer_,
                         self.like_hypers_,
                         self.basis_hypers_,
                         res.message))

        return self

    def _elbo(self, m, C, reg, lpars, bpars, X, y, *largs):
        # Full evidence lower bound objective with AEVB

        # Shapes
        D, K = m.shape

        # Make sure hypers and args can be unpacked into callables
        largs = tuple(chain(atleast_list(lpars), largs))

        # Basis function
        Phi = self.basis.transform(X, *atleast_list(bpars))  # M x D

        # Get regularizer
        L, slices = self.basis.regularizer_diagonal(X, *atleast_list(reg))
        iL = 1. / L[:, np.newaxis]

        # Posterior entropy lower bound terms
        logNkl = _qmatrix(m, C)
        logzk = logsumexp(logNkl, axis=0)

        # Preallocate variational parameter gradients and ELL
        dm = np.empty_like(m)
        dC = np.empty_like(C)
        Ell = np.empty(K, dtype=float)

        # Zero starts for sums over posterior mixtures
        dlpars = [np.zeros_like(p) for p in atleast_list(lpars)]
        EdPhi = np.zeros_like(Phi)

        # Log status, only do this occasionally to save cpu
        dolog = (self.__it % LOGITER == 0) or (self.__it == self.maxiter - 1)

        # Only calculate ELBO when sampling parameters (__it < 0) or logging
        calc_ll = dolog or (self.__it < 0)

        # Big loop though posterior mixtures for calculating stuff
        for k in range(K):

            # Sample expected likelihood and gradients
            Edmk, EdCk, EdPhik, Edlpars, Ell[k] = \
                self._reparam_k(m[:, k], C[:, k], y, Phi, largs, calc_ll)
            EdPhi += EdPhik / K

            # Weight factors for each component in the gradients
            Nkl_zk = np.exp(logNkl[:, k] - logzk[k])
            Nkl_zl = np.exp(logNkl[:, k] - logzk)
            alpha = (Nkl_zk + Nkl_zl)

            # Posterior mean and covariance gradients
            mkmj = m[:, k][:, np.newaxis] - m
            iCkCj = 1. / (C[:, k][:, np.newaxis] + C)

            dm[:, k] = (self.B_ * Edmk - m[:, k] / L
                        + (iCkCj * mkmj).dot(alpha)) / K
            dC[:, k] = (self.B_ * EdCk - 1. / L
                        + (iCkCj - (mkmj * iCkCj)**2).dot(alpha)) / (2 * K)

            # Likelihood parameter gradients
            for i, Edlpar in enumerate(Edlpars):
                dlpars[i] -= Edlpar / K

        # Regularizer gradient
        def dreg(s):
            dL = 0.5 * (((m[s]**2 + C[s]) * iL[s]**2).sum() / K - iL[s].sum())
            return -dL

        dL = list(map(dreg, slices)) if issequence(slices) else dreg(slices)

        # Basis function parameter gradients
        dtheta = lambda dPhi: -(EdPhi * dPhi).sum()
        dbpars = apply_grad(dtheta, self.basis.grad(X, *atleast_list(bpars)))

        # Approximate evidence lower bound
        ELBO = -np.inf
        if calc_ll:
            ELBO = (Ell.sum() * self.B_
                    - 0.5 * D * K * np.log(2 * np.pi)
                    - 0.5 * K * np.log(L).sum()
                    - 0.5 * ((m**2 + C) * iL).sum()
                    - logzk.sum() + np.log(K)) / K

        if dolog:
            rs_mes = "Random starts: " if self.__it < 0 else ""
            log.info("{}Iter {}: ELBO = {}, reg = {}, like_hypers = {}, "
                     "basis_hypers = {}"
                     .format(rs_mes, self.__it, ELBO, reg, lpars, bpars))

        self.__it += 1

        return -ELBO, [-dm, -dC, dL, dlpars, dbpars]

    def _reparam_k(self, mk, Ck, y, Phi, largs, calc_ll=True):
        # AEVB's reparameterisation trick

        # Sample the latent function and its derivative
        e = self.random_.randn(self.nsamples, len(mk))
        Sk = np.sqrt(Ck)
        ws = mk + Sk * e  # L x D
        fs = ws.dot(Phi.T)  # L x M
        dfs = self.likelihood.df(y, fs, *largs)  # L x M

        # Expected gradients
        Edws = dfs.dot(Phi)  # L x D, dweight samples
        Edm = Edws.sum(axis=0) / self.nsamples  # D
        EdC = (Edws * e / Sk).sum(axis=0) / self.nsamples  # D
        EdPhi = dfs.T.dot(ws) / self.nsamples  # M x D

        # Structured likelihood parameter gradients
        Edlpars = atleast_list(self.likelihood.dp(y, fs, *largs))
        for i, Edlpar in enumerate(Edlpars):
            Edlpars[i] = Edlpar.sum() / self.nsamples

        # Expected ll
        Ell = np.inf
        if calc_ll:
            Ell = self.likelihood.loglike(y, fs, *largs).sum() / self.nsamples

        return Edm, EdC, EdPhi, Edlpars, Ell

    def predict(self, X, nsamples=200, likelihood_args=()):
        """
        Predict target values from Bayesian generalized linear regression.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, d dimensions).
        nsamples : int, optional
            Number of samples for sampling the expected target values from the
            predictive distribution.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N.

        Returns
        -------
        Ey : ndarray
            The expected value of y* for the query inputs, X* of shape (N*,).
        """
        Ey, _ = self.predict_moments(X, nsamples, likelihood_args)

        return Ey

    def predict_moments(self, X, nsamples=200, likelihood_args=()):
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
        distribution over weights (from ``learn``). Here are few concrete
        examples of how we can use these values,

        - Gaussian likelihood: these are just the predicted mean and variance,
          see ``revrand.regression.predict``
        - Bernoulli likelihood: The expected value is the probability,
          :math:`p(y^* = 1)`, i.e. the probability of class one. The variance
          may not be so useful.
        - Poisson likelihood: The expected value is similar conceptually to the
          Gaussian case, and is also a *continuous* value. The median (50%
          quantile) from ``predict_interval`` is a discrete value. Again,
          the variance in this instance may not be so useful.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, d
            dimensions).
        nsamples : int, optional
            Number of samples for sampling the expected moments from the
            predictive distribution.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N.

        Returns
        -------
        Ey : ndarray
            The expected value of y* for the query inputs, X* of shape (N*,).
        Vy : ndarray
            The expected variance of y* (excluding likelihood noise terms) for
            the query inputs, X* of shape (N*,).
        """
        # Get latent function samples
        N = X.shape[0]
        ys = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood expected value
        Eyargs = tuple(chain(atleast_list(self.like_hypers_), likelihood_args))
        for i, f in enumerate(fsamples):
            ys[:, i] = self.likelihood.Ey(f, *Eyargs)

        # Average transformed samples (MC integration)
        Ey = ys.mean(axis=1)
        Vy = ((ys - Ey[:, np.newaxis])**2).mean(axis=1)

        return Ey, Vy

    def predict_logpdf(self, X, y, nsamples=200, likelihood_args=()):
        r"""
        Predictive log-probability density function of a Bayesian GLM.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, D dimensions).
        y : float or ndarray
            The test observations of shape (N*,) to evaluate under,
            :math:`\log p(y^* |\mathbf{x}^*, \mathbf{X}, y)`.
        nsamples : int, optional
            Number of samples for sampling the log predictive distribution.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N*.

        Returns
        -------
        logp : ndarray
           The log probability of y* given X* of shape (N*,).
        logp_min : ndarray
            The minimum sampled values of the predicted log probability (same
            shape as p)
        logp_max : ndarray
            The maximum sampled values of the predicted log probability (same
            shape as p)
        """
        X, y = check_X_y(X, y)

        # Get latent function samples
        N = X.shape[0]
        ps = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood pdf
        llargs = tuple(chain(atleast_list(self.like_hypers_), likelihood_args))
        for i, f in enumerate(fsamples):
            ps[:, i] = self.likelihood.loglike(y, f, *llargs)

        # Average transformed samples (MC integration)
        logp = ps.mean(axis=1)
        logp_min = ps.min(axis=1)
        logp_max = ps.max(axis=1)

        return logp, logp_min, logp_max

    def predict_cdf(self, X, quantile, nsamples=200, likelihood_args=()):
        r"""
        Predictive cumulative density function of a Bayesian GLM.

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, D dimensions).
        quantile : float
            The predictive probability, :math:`p(y^* \leq \text{quantile} |
            \mathbf{x}^*, \mathbf{X}, y)`.
        nsamples : int, optional
            Number of samples for sampling the predictive CDF.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N*.
        nsamples : int, optional
            The number of samples to draw from the posterior in order to
            approximate the predictive mean and variance.

        Returns
        -------
        p : ndarray
           The probability of y* <= quantile for the query inputs, X* of shape
           (N*,).
        p_min : ndarray
            The minimum sampled values of the predicted probability (same shape
            as p)
        p_max : ndarray
            The maximum sampled values of the predicted probability (same shape
            as p)
        """
        # Get latent function samples
        N = X.shape[0]
        ps = np.empty((N, nsamples))
        fsamples = self._sample_func(X, nsamples)

        # Push samples though likelihood cdf
        cdfarg = tuple(chain(atleast_list(self.like_hypers_), likelihood_args))
        for i, f in enumerate(fsamples):
            ps[:, i] = self.likelihood.cdf(quantile, f, *cdfarg)

        # Average transformed samples (MC integration)
        p = ps.mean(axis=1)
        p_min = ps.min(axis=1)
        p_max = ps.max(axis=1)

        return p, p_min, p_max

    def predict_interval(self, X, percentile, nsamples=200, likelihood_args=(),
                         multiproc=True):
        """
        Predictive percentile interval (upper and lower quantiles).

        Parameters
        ----------
        X : ndarray
            (N*,d) array query input dataset (N* samples, D dimensions).
        percentile : float
            The percentile confidence interval (e.g. 95%) to return.
        nsamples : int, optional
            Number of samples for sampling the predictive percentiles.
        likelihood_args : sequence, optional
            sequence of arguments to pass to the likelihood function. These are
            non-learnable parameters. They can be scalars or arrays of length
            N*.
        multiproc : bool, optional
            Use multiprocessing to paralellise this prediction computation.

        Returns
        -------
        ql : ndarray
            The lower end point of the interval with shape (N*,)
        qu : ndarray
            The upper end point of the interval with shape (N*,)
        """
        N = X.shape[0]

        # Generate latent function samples per observation (n in N)
        fsamples = self._sample_func(X, nsamples, genaxis=0)

        # Make sure likelihood_args is consistent with work
        if len(likelihood_args) > 0:
            likelihood_args = _reshape_likelihood_args(likelihood_args, N)

        # Now create work for distrbuted workers
        like_hypers = atleast_list(self.like_hypers_)
        work = ((f[0], self.likelihood, like_hypers, f[1:], percentile)
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

    def _sample_func(self, X, nsamples, genaxis=1):
        """
        Generate samples from the posterior latent function mixtures of the GLM
        for query inputs, X*.

        Parameters
        ----------
        X : ndarray
            (N*, d) array query input dataset (N* samples, D dimensions).
        nsamples : int
            Number of samples for sampling the latent function.
        genaxis : int
            Axis to return samples from, i.e.
            - ``genaxis=1`` will give you one sample at a time of f for ALL
                observations (so it will iterate over nsamples).
            - ``genaxis=0`` will give you all samples of f for ONE
                observation at a time (so it will iterate through X*, row by
                row)

        Yields
        ------
        fsamples : ndarray
            of shape (N*,) if ``genaxis=1`` with each call being a sample
            from the mixture of latent functions over all N*. Or of shape
            (nsamples,) if ``genaxis=0``, with each call being a all
            samples for an observation, n in N*.
        """
        check_is_fitted(self, ['weights_', 'covariance_', 'basis_hypers_',
                               'like_hypers_', 'regularizer_'])
        X = check_array(X)
        D, K = self.weights_.shape

        # Generate weight samples from all mixture components
        k = self.random_.randint(0, K, size=(nsamples,))
        w = self.weights_[:, k] + self.random_.randn(D, nsamples) \
            * np.sqrt(self.covariance_[:, k])

        # Do this here for *massive* speed improvements
        Phi = self.basis.transform(X, *atleast_list(self.basis_hypers_))

        # Now generate latent functions samples either colwise or rowwise
        if genaxis == 1:
            fs = (Phi.dot(ws) for ws in w.T)
        elif genaxis == 0:
            fs = (phi_n.dot(w) for phi_n in Phi)
        else:
            raise ValueError("Invalid axis to generate samples from")

        return fs

    def __repr__(self):
        """Representation."""
        return "{}(likelihood={}, basis={}, K={}, maxiter={}, batch_size={},"\
            "updater={}, nsamples={}, nstarts={}, random_state={})".format(
                self.__class__.__name__,
                self.likelihood,
                self.basis,
                self.K,
                self.maxiter,
                self.batch_size,
                self.updater,
                self.nsamples,
                self.nstarts,
                self.random_state
            )


# For GB/AU spelling
class GeneralisedLinearModel(GeneralizedLinearModel):
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


def _qmatrix(m, C):

    K = m.shape[1]
    logq = [[_dgausll(m[:, i], m[:, j], C[:, i] + C[:, j])
             for i in range(K)]
            for j in range(K)]

    return np.array(logq)
