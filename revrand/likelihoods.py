"""Likelihood objects for inference within the GLM framework."""

from __future__ import division

import numpy as np

from scipy.stats import bernoulli, binom, poisson, norm
from scipy.special import gammaln, expit

from .btypes import Parameter, Positive
from .mathfun.special import safesoftplus, softplus


#
# Likelihood classes
#

class Bernoulli():
    r"""
    Bernoulli likelihood class for (binary) classification tasks.

    A logistic transformation function is used to map the latent function from
    the GLM prior into a probability.

    .. math::

        p(y_i | f_i) = \sigma(f_i) ^ {y_i} (1 - \sigma(f_i))^{1 - y_i}

    where :math:`y_i` is a target, :math:`f_i` the value of the latent function
    corresponding to the target, and :math:`\sigma(\cdot)` is the logistic
    sigmoid.
    """

    _params = Parameter()

    def __init__(self):

        pass

    @property
    def params(self):
        """Get this object's Parameter types."""
        return self._params

    @params.setter
    def params(self, params):
        """Set this object's Parameter types."""
        self._params = params

    def loglike(self, y, f):
        r"""
        Bernoulli log likelihood.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        logp: ndarray
            the log likelihood of each y given each f under this
            likelihood.
        """
        # way faster than calling bernoulli.logpmf
        y, f = np.broadcast_arrays(y, f)
        ll = y * f - softplus(f)
        return ll

    def Ey(self, f):
        r"""
        Expected value of the Bernoulli likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[\mathbf{y}|\mathbf{f}]`.
        """
        return expit(f)

    def df(self, y, f):
        r"""
        Derivative of Bernoulli log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        df: ndarray
            the derivative :math:`\partial \log p(y|f) / \partial f`
        """
        y, f = np.broadcast_arrays(y, f)
        return y - expit(f)

    def dp(self, y, f, *args):
        r"""
        Derivative of Bernoulli log likelihood w.r.t.\  the parameters,
        :math:`\theta`.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        dp: list, float or ndarray
            the derivative
            :math:`\partial \log p(y|f, \theta)/ \partial \theta` for
            each parameter. If there is only one parameter, this is not a
            list.
        """
        return []

    def cdf(self, y, f):
        r"""
        Cumulative density function of the likelihood.

        Parameters
        ----------
        y: ndarray
            query quantiles, i.e.\  :math:`P(Y \leq y)`.
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        cdf: ndarray
            Cumulative density function evaluated at y.
        """
        return bernoulli.cdf(y, expit(f))


class Binomial(Bernoulli):
    r"""
    Binomial likelihood class.

    A logistic transformation function is used to map the latent function from
    the GLM prior into a probability.

    .. math::

        p(y_i | f_i) = \genfrac(){0pt}{}{n}{y_i}
            \sigma(f_i) ^ {y_i} (1 - \sigma(f_i))^{n - y_i}

    where :math:`y_i` is a target, :math:`f_i` the value of the latent function
    corresponding to the target, :math:`n` is the total possible count, and
    :math:`\sigma(\cdot)` is the logistic sigmoid. :math:`n` can also be
    applied per observation.
    """

    def loglike(self, y, f, n):
        r"""
        Binomial log likelihood.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        n: ndarray
            the total number of observations

        Returns
        -------
        logp: ndarray
            the log likelihood of each y given each f under this
            likelihood.
        """
        ll = binom.logpmf(y, n=n, p=expit(f))
        return ll

    def Ey(self, f, n):
        r"""
        Expected value of the Binomial likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        n: ndarray
            the total number of observations

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[\mathbf{y}|\mathbf{f}]`.
        """
        return expit(f) * n

    def df(self, y, f, n):
        r"""
        Derivative of Binomial log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        n: ndarray
            the total number of observations

        Returns
        -------
        df: ndarray
            the derivative :math:`\partial \log p(y|f) / \partial f`
        """
        y, f, n = np.broadcast_arrays(y, f, n)
        return y - expit(f) * n

    def cdf(self, y, f, n):
        r"""
        Cumulative density function of the likelihood.

        Parameters
        ----------
        y: ndarray
            query quantiles, i.e.\  :math:`P(Y \leq y)`.
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        n: ndarray
            the total number of observations

        Returns
        -------
        cdf: ndarray
            Cumulative density function evaluated at y.
        """
        return binom.cdf(y, n=n, p=expit(f))


class Gaussian(Bernoulli):
    r"""
    A univariate Gaussian likelihood for general regression tasks.

    No transformation function is needed since this is (conditionally)
    conjugate to the GLM prior.

    .. math::

        p(y_i | f_i) = \frac{1}{\sqrt{2 \pi \sigma^2}}
            \exp\left(- \frac{(y_i - f_i)^2}{2 \sigma^2} \right)

    where :math:`y_i` is a target, :math:`f_i` the value of the latent function
    corresponding to the target and :math:`\sigma` is the observation noise
    (standard deviation).

    Parameters
    ----------
    var_init: Parameter, optional
        A scalar Parameter describing the initial point and bounds for
        an optimiser to learn the variance parameter of this object.
    """

    def __init__(self, var_init=Parameter(1., Positive())):

        self.params = var_init

    def loglike(self, y, f, var):
        r"""
        Gaussian log likelihood.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        var: float, ndarray
            The variance of the distribution

        Returns
        -------
        logp: ndarray
            the log likelihood of each y given each f under this
            likelihood.
        """
        # way faster than calling norm.logpdf
        y, f = np.broadcast_arrays(y, f)
        ll = - 0.5 * (np.log(2 * np.pi * var) + (y - f)**2 / var)
        return ll

    def Ey(self, f, var):
        r"""
        Expected value of the Gaussian likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        var: float, ndarray
            The variance of the distribution

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[\mathbf{y}|\mathbf{f}]`.
        """
        return f

    def df(self, y, f, var):
        r"""
        Derivative of Gaussian log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        var: float, ndarray
            The variance of the distribution

        Returns
        -------
        df: ndarray
            the derivative :math:`\partial \log p(y|f) / \partial f`
        """
        y, f = np.broadcast_arrays(y, f)
        return (y - f) / var

    def dp(self, y, f, var):
        r"""
        Derivative of Gaussian log likelihood w.r.t.\ the variance
        :math:`\sigma^2`.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        var: float, ndarray
            The variance of the distribution

        Returns
        -------
        dp: float
            the derivative
            :math:`\partial \log p(y|f, \sigma^2)/ \partial \sigma^2`
            where :math:`sigma^2` is the variance.
        """
        y, f = np.broadcast_arrays(y, f)
        ivar = 1. / var
        return 0.5 * (((y - f) * ivar)**2 - ivar)

    def cdf(self, y, f, var):
        r"""
        Cumulative density function of the likelihood.

        Parameters
        ----------
        y: ndarray
            query quantiles, i.e.\  :math:`P(Y \leq y)`.
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)
        var: float, ndarray
            The variance of the distribution

        Returns
        -------
        cdf: ndarray
            Cumulative density function evaluated at y.
        """
        return norm.cdf(y, loc=f, scale=np.sqrt(var))


class Poisson(Bernoulli):
    r"""
    A Poisson likelihood, useful for various Poisson process tasks.

    An exponential transformation function and a softplus transformation
    function have been implemented.

    .. math::

        p(y_i | f_i) = \frac{g(f_i)^{y_i} e^{-g(f_i)}}{y_i!}

    where :math:`y_i` is a target, :math:`f_i` the value of the latent function
    corresponding to the target, and :math:`g(\cdot)` is the tranformation
    function, which can be either an exponential function, or a softplus
    function (:math:`\log(1 + \exp(f_i)`).

    Parameters
    ----------
    tranfcn: string, optional
        this may be 'exp' for an exponential transformation function,
        or 'softplus' for a softplut transformation function.
    """

    def __init__(self, tranfcn='exp'):

        if tranfcn == 'exp' or tranfcn == 'softplus':
            self.tranfcn = tranfcn
        else:
            raise ValueError('Invalid transformation function specified!')

    def loglike(self, y, f):
        r"""
        Poisson log likelihood.

        Parameters
        ----------
        y: ndarray
            array of integer targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        logp: ndarray
            the log likelihood of each y given each f under this
            likelihood.
        """
        y, f = np.broadcast_arrays(y, f)
        if self.tranfcn == 'exp':
            g = np.exp(f)
            logg = f
        else:
            g = softplus(f)
            logg = np.log(g)
        return y * logg - g - gammaln(y + 1)

    def Ey(self, f):
        r"""
        Expected value of the Poisson likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[\mathbf{y}|\mathbf{f}]`.
        """
        return np.exp(f) if self.tranfcn == 'exp' else softplus(f)

    def df(self, y, f):
        r"""
        Derivative of Poisson log likelihood w.r.t.\  f.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        df: ndarray
            the derivative :math:`\partial \log p(y|f) / \partial f`
        """
        y, f = np.broadcast_arrays(y, f)
        if self.tranfcn == 'exp':
            return y - np.exp(f)
        else:
            return expit(f) * (y / safesoftplus(f) - 1)

    def cdf(self, y, f):
        r"""
        Cumulative density function of the likelihood.

        Parameters
        ----------
        y: ndarray
            query quantiles, i.e.\  :math:`P(Y \leq y)`.
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        cdf: ndarray
            Cumulative density function evaluated at y.
        """
        mu = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        return poisson.cdf(y, mu=mu)
