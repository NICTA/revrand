"""
Likelihood objects for inference within the GLM framework.

"""

from __future__ import division

import numpy as np

from scipy.stats import bernoulli, binom, poisson, norm
from scipy.special import gammaln, expit

from .btypes import Parameter, Positive
from .mathfun.special import safesoftplus, softplus, logtiny


#
# Likelihood classes
#

class Bernoulli():
    """
    Bernoulli likelihood class for (binary) classification tasks.

    A logistic transformation function is used to map the latent function from
    the GLM prior into a probability.
    """

    _params = []

    def __init__(self):

        pass

    @property
    def params(self):
        """ Get this object's Parameter types. """
        return self._params

    @params.setter
    def params(self, params):
        """ Set this object's Parameter types. """
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

        ll = bernoulli.logpmf(y, expit(f))
        ll[np.isinf(ll)] = logtiny
        return ll

    def Ey(self, f):
        r""" Expected value of the Bernoulli likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[y|f]`.
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

        return y - expit(f)

    def d2f(self, y, f):
        r"""
        Second derivative of Bernoulli log likelihood w.r.t.\  f.

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
            the second derivative
            :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        sig = expit(f)
        return (sig - 1) * sig

    def d3f(self, y, f):
        r"""
        Third derivative of Bernoulli log likelihood w.r.t.\  f.

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
            the third derivative
            :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        sig = expit(f)
        return (2 * sig - 1) * (1 - sig) * sig

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

    def dpd2f(self, y, f, *args):
        r"""
        Partial derivative of Bernoulli log likelihood,
        :math:`\partial h(f, \theta) / \partial \theta` where
        :math:`h(f, \theta) = \partial^2 \log p(y|f, \theta)/ \partial f^2`.

        Parameters
        ----------
        y: ndarray
            array of 0, 1 valued integers of targets
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        dpd2f: list or ndarray
            the derivative of the likelihood Hessian w.r.t.\
            :math:`\theta` for each parameter. If there is only one
            parameter, this is not a list.
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
    """
    Binomial likelihood class.

    A logistic transformation function is used to map the latent function from
    the GLM prior into a probability.
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
        ll[np.isinf(ll)] = logtiny
        return ll

    def Ey(self, f, n):
        r""" Expected value of the Binomial likelihood.

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
            expected value of y, :math:`\mathbb{E}[y|f]`.
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

        return y - expit(f) * n

    def d2f(self, y, f, n):
        r"""
        Second derivative of Binomial log likelihood w.r.t.\  f.

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
            the second derivative
            :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        sig = expit(f)
        return (sig - 1) * sig * n

    def d3f(self, y, f, n):
        r"""
        Third derivative of Binomial log likelihood w.r.t.\  f.

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
            the third derivative
            :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        sig = expit(f)
        return (2 * sig - 1) * (1 - sig) * sig * n

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
    """
    A univariate Gaussian likelihood for general regression tasks.

    No transformation function is needed since this is (conditionally)
    conjugate to the GLM prior.

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

        return norm.logpdf(y, loc=f, scale=np.sqrt(var))

    def Ey(self, f, var):
        r""" Expected value of the Gaussian likelihood.

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
            expected value of y, :math:`\mathbb{E}[y|f]`.
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

        return (y - f) / var

    def d2f(self, y, f, var):
        r"""
        Second derivative of Gaussian log likelihood w.r.t.\  f.

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
            the second derivative
            :math:`\partial^2 \log p(y|f)/ \partial f^2`
        """

        return - np.ones_like(f) / var

    def d3f(self, y, f, var):
        r"""
        Third derivative of Gaussian log likelihood w.r.t.\  f.

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
            the third derivative
            :math:`\partial^3 \log p(y|f)/ \partial f^3`
        """

        return np.zeros_like(f)

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

        ivar = 1. / var
        return 0.5 * (((y - f) * ivar)**2 - ivar)

    def dpd2f(self, y, f, var):
        r"""
        Partial derivative of Gaussian log likelihood,
        :math:`\partial h(f, \theta) / \partial \theta` where
        :math:`h(f, \theta) = \partial^2 \log p(y|f, \theta)/ \partial f^2`.

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
        dpd2f: ndarray
            the derivative of the likelihood Hessian w.r.t.\ the variance
            :math:`\sigma^2`.
        """

        return np.ones_like(f) / var**2

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
    """
    A Poisson likelihood, useful for various Poisson process tasks.

    An exponential transformation function and a softplus transformation
    function have been implemented.

    Parameters
    ----------
    tranfcn: string, optional
        this may be 'exp' for an exponential transformation function,
        or 'softplus' for a softplut transformation function.
    """

    def __init__(self, tranfcn='exp'):
        """
        Construct an instance of the Poisson likelihood class.


        """

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

        g = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        logg = np.log(g)
        if not np.isscalar(logg):
            logg[np.isinf(logg)] = logtiny
        else:
            logg = logtiny if np.isinf(logg) else logg
        return y * logg - g - gammaln(y + 1)

    def Ey(self, f):
        r""" Expected value of the Poisson likelihood.

        Parameters
        ----------
        f: ndarray
            latent function from the GLM prior (:math:`\mathbf{f} =
            \boldsymbol\Phi \mathbf{w}`)

        Returns
        -------
        Ey: ndarray
            expected value of y, :math:`\mathbb{E}[y|f]`.
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

        if self.tranfcn == 'exp':
            return y - np.exp(f)
        else:
            return expit(f) * (y / safesoftplus(f) - 1)

    def d2f(self, y, f):
        r"""
        Second derivative of Poisson log likelihood w.r.t.\  f.

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
        r"""
        Third derivative of Poisson log likelihood w.r.t.\  f.

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
