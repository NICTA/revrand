"""
Likelihood objects for inference within the GLM framework.

"""

from __future__ import division

import numpy as np
from scipy.stats import bernoulli, poisson, norm
from scipy.special import gammaln

from .utils import Positive
from .transforms import logistic, softplus


#
# Module constants
#

tiny = np.finfo(float).tiny
logtiny = np.log(tiny)
small = 1e-100
resol = np.finfo(float).resolution


#
# Likelihood classes
#

class Bernoulli():
    """
    Bernoulli likelihood class for (binary) classification tasks.

    A logistic transformation function is used to map the latent function from
    the GLM prior into a probability.
    """

    _bounds = []

    def __init__(self):

        pass

    @property
    def bounds(self):
        """
        Get this object's parameter bounds. This is a list of pairs of upper
        and lower bounds, with the same length as the total number of scalars
        in all of the parameters combined (and in order).
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """
        Set this object's parameter bounds. This is a list of pairs of upper
        and lower bounds, with the same length as the total number of scalars
        in all of the parameters combined (and in order).
        """
        self._bounds = bounds

    def loglike(self, y, f):
        """
        Bernoulli log likelihood.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            logp: array_like
                the log likelihood of each y given each f under this
                likelihood.
        """

        ll = bernoulli.logpmf(y, logistic(f))
        ll[np.isinf(ll)] = logtiny
        return ll

    def Ey(self, f):
        """ Expected value of the Bernoulli likelihood.

        Parameters
        ----------
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            Ey: array_like
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return logistic(f)

    def df(self, y, f):
        """
        Derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        return y - logistic(f)

    def d2f(self, y, f):
        """
        Second derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        sig = logistic(f)
        return (sig - 1) * sig

    def d3f(self, y, f):
        """
        Third derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the third derivative
                :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        sig = logistic(f)
        return (2 * sig - 1) * (1 - sig) * sig

    def dp(self, y, f):
        """
        Derivative of Bernoulli log likelihood w.r.t.\  the parameters,
        :math:`\\theta`.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            dp: list
                the derivative
                :math:`\partial \log p(y|f, \\theta)/ \partial \\theta` for
                each parameter.
        """

        return []

    def dpd2f(self, y, f):
        """
        Partial derivative of Bernoulli log likelihood,
        :math:`\partial h(f, \\theta) / \partial \\theta` where
        :math:`h(f, \\theta) = \partial^2 \log p(y|f, \\theta)/ \partial f^2`.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            dp: list
                the derivative of the likelihood Hessian w.r.t.\
                :math:`\\theta` for each parameter.
        """

        return []

    def cdf(self, y, f):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: array_like
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            cdf: ndarray
                Cumulative density function evaluated at y.
        """

        return bernoulli.cdf(y, logistic(f))


class Gaussian(Bernoulli):
    """
    A univariate Gaussian likelihood for general regression tasks.

    No transformation function is needed since this is conjugate to the GLM
    prior.
    """

    def __init__(self, var_bounds=Positive()):
        """
        Construct an instance of the Gaussian likelihood class.

        Parameters
        ----------
            var_bounds: tuple, optional
                A tuple of (upper, lower) bounds of the variance.
        """

        self.bounds = [var_bounds]

    def loglike(self, y, f, var):
        """
        Gaussian log likelihood.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            logp: array_like
                the log likelihood of each y given each f under this
                likelihood.
        """

        return norm.logpdf(y, loc=f, scale=np.sqrt(var))

    def Ey(self, f, var):
        """ Expected value of the Gaussian likelihood.

        Parameters
        ----------
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            Ey: array_like
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return f

    def df(self, y, f, var):
        """
        Derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            df: array_like
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        return (y - f) / var

    def d2f(self, y, f, var):
        """
        Second derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            df: array_like
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        return - np.ones_like(f) / var

    def d3f(self, y, f, var):
        """
        Third derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            df: array_like
                the third derivative
                :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        return np.zeros_like(f)

    def dp(self, y, f, var):
        """
        Derivative of Gaussian log likelihood w.r.t.\  the parameters,
        :math:`\\theta`.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            dp: list
                the derivative
                :math:`\partial \log p(y|f, \\theta)/ \partial \\theta` for
                each parameter.
        """

        return [0.5 * (((y - f) / var)**2 - 1. / var)]

    def dpd2f(self, y, f, var):
        """
        Partial derivative of Gaussian log likelihood,
        :math:`\partial h(f, \\theta) / \partial \\theta` where
        :math:`h(f, \\theta) = \partial^2 \log p(y|f, \\theta)/ \partial f^2`.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            dp: list
                the derivative of the likelihood Hessian w.r.t.\
                :math:`\\theta` for each parameter.
        """

        return [np.ones_like(f) / var**2]

    def cdf(self, y, f, var):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: array_like
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: array_like
                latent function from the GLM prior
            var: float, array_like
                The variance of the distribution

        Returns
        -------
            cdf: ndarray
                Cumulative density function evaluated at y.
        """

        return norm.cdf(y, loc=f, scale=np.sqrt(var))


class Poisson(Bernoulli):
    """
    A Poisson likelihood, useful for various Poisson process tasks.

    An exponential transformation function and a softplus transformation
    function have been implemented.

    """

    def __init__(self, tranfcn='exp'):
        """
        Construct an instance of the Poisson likelihood class.

        Parameters
        ----------
            tranfcn: string, optional
                this may be 'exp' for an exponential transformation function,
                or 'softplus' for a softplut transformation function.
        """

        if tranfcn == 'exp' or tranfcn == 'softplus':
            self.tranfcn = tranfcn
        else:
            raise ValueError('Invalid transformation function specified!')

    def loglike(self, y, f):
        """
        Poisson log likelihood.

        Parameters
        ----------
            y: array_like
                array of integer targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            logp: array_like
                the log likelihood of each y given each f under this
                likelihood.
        """

        g = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        logg = np.log(g)
        logg[np.isinf(logg)] = logtiny
        return y * logg - g - gammaln(y + 1)

    def Ey(self, f):
        """ Expected value of the Poisson likelihood.

        Parameters
        ----------
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            Ey: array_like
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return np.exp(f) if self.tranfcn == 'exp' else softplus(f)

    def df(self, y, f):
        """
        Derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        if self.tranfcn == 'exp':
            return y - np.exp(f)
        else:
            return logistic(f) * (y / _safesoftplus(f) - 1)

    def d2f(self, y, f):
        """
        Second derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        if self.tranfcn == 'exp':
            return - np.exp(f)
        else:
            g = _safesoftplus(f)
            gp = logistic(f)
            g2p = gp * (1 - gp)
            return (y - g) * g2p / g - y * (gp / g)**2

    def d3f(self, y, f):
        """
        Third derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: array_like
                array of 0, 1 valued integers of targets
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            df: array_like
                the third derivative
                :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        if self.tranfcn == 'exp':
            return self.d2f(y, f)
        else:
            g = _safesoftplus(f)
            gp = logistic(f)
            g2p = gp * (1 - gp)
            g3p = g2p * (1 - 2 * gp)
            return g3p * (y - g) / g + 2 * y * (gp / g)**3 \
                - 3 * y * gp * g3p / g**2

    def cdf(self, y, f):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: array_like
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: array_like
                latent function from the GLM prior

        Returns
        -------
            cdf: ndarray
                Cumulative density function evaluated at y.
        """

        mu = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        return poisson.cdf(y, mu=mu)


#
# Safe numerical operations
#

def _safelog(x):

    cx = x.copy()
    cx[cx < tiny] = tiny
    return np.log(cx)


def _safesoftplus(x):

    g = softplus(x)
    g[g < small] = small  # hack to avoid instability
    return g
