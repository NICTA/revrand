"""
Likelihood objects for inference within the GLM framework.

"""

from __future__ import division

import numpy as np
from scipy.stats import bernoulli, poisson, norm
from scipy.special import gammaln, expit

from .optimize import Positive
from .utils import safesoftplus, safediv, softplus


#
# Module constants
#

tiny = np.finfo(float).tiny
logtiny = np.log(tiny)


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
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            logp: ndarray
                the log likelihood of each y given each f under this
                likelihood.
        """

        ll = bernoulli.logpmf(y, expit(f))
        ll[np.isinf(ll)] = logtiny
        return ll

    def Ey(self, f):
        """ Expected value of the Bernoulli likelihood.

        Parameters
        ----------
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            Ey: ndarray
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return expit(f)

    def df(self, y, f):
        """
        Derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        return y - expit(f)

    def d2f(self, y, f):
        """
        Second derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        sig = expit(f)
        return (sig - 1) * sig

    def d3f(self, y, f):
        """
        Third derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the third derivative
                :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        sig = expit(f)
        return (2 * sig - 1) * (1 - sig) * sig

    def dp(self, y, f):
        """
        Derivative of Bernoulli log likelihood w.r.t.\  the parameters,
        :math:`\\theta`.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            dp: list, float or ndarray
                the derivative
                :math:`\partial \log p(y|f, \\theta)/ \partial \\theta` for
                each parameter. If there is only one parameter, this is not a
                list.
        """

        return []

    def dpd2f(self, y, f):
        """
        Partial derivative of Bernoulli log likelihood,
        :math:`\partial h(f, \\theta) / \partial \\theta` where
        :math:`h(f, \\theta) = \partial^2 \log p(y|f, \\theta)/ \partial f^2`.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            dpd2f: list or ndarray
                the derivative of the likelihood Hessian w.r.t.\
                :math:`\\theta` for each parameter. If there is only one
                parameter, this is not a list.
        """

        return []

    def cdf(self, y, f):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: ndarray
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            cdf: ndarray
                Cumulative density function evaluated at y.
        """

        return bernoulli.cdf(y, expit(f))


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
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            logp: ndarray
                the log likelihood of each y given each f under this
                likelihood.
        """

        return norm.logpdf(y, loc=f, scale=np.sqrt(var))

    def Ey(self, f, var):
        """ Expected value of the Gaussian likelihood.

        Parameters
        ----------
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            Ey: ndarray
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return f

    def df(self, y, f, var):
        """
        Derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            df: ndarray
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        return safediv(y - f, var)

    def d2f(self, y, f, var):
        """
        Second derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            df: ndarray
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        return - safediv(np.ones_like(f), var)

    def d3f(self, y, f, var):
        """
        Third derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            df: ndarray
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
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            dp: float
                the derivative
                :math:`\partial \log p(y|f, \\sigma^2)/ \partial \\sigma^2`
                where :math:`sigma^2` is the variance.
        """

        return 0.5 * (safediv(y - f, var)**2 - safediv(1., var))

    def dpd2f(self, y, f, var):
        """
        Partial derivative of Gaussian log likelihood,
        :math:`\partial h(f, \\theta) / \partial \\theta` where
        :math:`h(f, \\theta) = \partial^2 \log p(y|f, \\theta)/ \partial f^2`.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
                The variance of the distribution

        Returns
        -------
            dpd2f: ndarray
                the derivative of the likelihood Hessian w.r.t.\ the variance
                :math:`\\sigma^2`.
        """

        return safediv(np.ones_like(f), var**2)

    def cdf(self, y, f, var):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: ndarray
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)
            var: float, ndarray
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
            y: ndarray
                array of integer targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            logp: ndarray
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
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            Ey: ndarray
                expected value of y, :math:`\mathbb{E}[y|f]`.
        """

        return np.exp(f) if self.tranfcn == 'exp' else softplus(f)

    def df(self, y, f):
        """
        Derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the derivative :math:`\partial \log p(y|f) / \partial f`
        """

        if self.tranfcn == 'exp':
            return y - np.exp(f)
        else:
            return expit(f) * (y / safesoftplus(f) - 1)

    def d2f(self, y, f):
        """
        Second derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the second derivative
                :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        if self.tranfcn == 'exp':
            return - np.exp(f)
        else:
            g = safesoftplus(f)
            gp = expit(f)
            g2p = gp * (1 - gp)
            return (y - g) * g2p / g - y * (gp / g)**2

    def d3f(self, y, f):
        """
        Third derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
            y: ndarray
                array of 0, 1 valued integers of targets
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            df: ndarray
                the third derivative
                :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        if self.tranfcn == 'exp':
            return self.d2f(y, f)
        else:
            g = safesoftplus(f)
            gp = expit(f)
            g2p = gp * (1 - gp)
            g3p = g2p * (1 - 2 * gp)
            return g3p * (y - g) / g + 2 * y * (gp / g)**3 \
                - 3 * y * gp * g3p / g**2

    def cdf(self, y, f):
        """
        Cumulative density function of the likelihood.

        Parameters
        ----------
            y: ndarray
                query quantiles, i.e.\  :math:`P(Y \leq y)`.
            f: ndarray
                latent function from the GLM prior (:math:`\mathbf{f} =
                \\boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
            cdf: ndarray
                Cumulative density function evaluated at y.
        """

        mu = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        return poisson.cdf(y, mu=mu)
