"""
Likelihood objects for inference within the GLM framework.
"""

import numpy as np
# from scipy.special import gammaln
from scipy.stats import bernoulli, poisson, norm
from .utils import Positive
from .transforms import logistic, softplus


# Module constants
tiny = np.finfo(float).tiny
small = 1e-100
resol = np.finfo(float).resolution


#
# Likelihood classes
#

class Bernoulli():

    _bounds = []

    def __init__(self):

        pass

    @property
    def bounds(self):
        """
        Get this object's parameter bounds. This is a list of pairs of upper
        and lower bounds, with the same length as the total number of scalars
        in all of the (non-transformation) parameters combined (and in order).
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """
        Set this object's parameter bounds. This is a list of pairs of upper
        and lower bounds, with the same length as the total number of scalars
        in all of the (non-transformation) parameters combined (and in order).
        """
        self._bounds = bounds

    def loglike(self, y, f):

        return bernoulli.logpmf(y, logistic(f))

    def Ey(self, f):

        return logistic(f)

    def df(self, y, f):

        return y - logistic(f)

    def d2f(self, y, f):

        sig = logistic(f)
        return (sig - 1) * sig

    def d3f(self, y, f):

        sig = logistic(f)
        return (2 * sig - 1) * (1 - sig) * sig

    def dp(self, y, f):

        return []

    def dpd2f(self, y, f):

        return []

    def cdf(self, y, f):

        return bernoulli.cdf(y, logistic(f))

    def interval(self, alpha, f):

        return bernoulli.interval(alpha, logistic(f))


class Gaussian(Bernoulli):

    def __init__(self, var_bounds=Positive()):

        self.bounds = [var_bounds]

    def loglike(self, y, f, var):

        return norm.logpdf(y, loc=f, scale=np.sqrt(var))

    def Ey(self, f, var):

        return f

    def df(self, y, f, var):

        return (y - f) / var

    def d2f(self, y, f, var):

        return - np.ones_like(f) / var

    def d3f(self, y, f, var):

        return np.zeros_like(f)

    def dp(self, y, f, var):

        return [0.5 * (((y - f) / var)**2 - 1. / var)]

    def dpd2f(self, y, f, var):

        return [np.ones_like(f) / var**2]

    def cdf(self, y, f, var):

        return norm.cdf(y, logistic(f), scale=np.sqrt(var))

    def interval(self, alpha, f, var):

        return norm.interval(alpha, loc=f, scale=np.sqrt(var))


class Poisson(Bernoulli):

    def __init__(self, tranfcn='softplus'):

        if tranfcn == 'exp' or tranfcn == 'softplus':
            self.tranfcn = tranfcn
        else:
            raise ValueError('Invalid transformation function specified!')

    def loglike(self, y, f):

        g = np.exp(f) if self.tranfcn == 'exp' else softplus(f)
        return poisson.logpmf(y, g)

    def Ey(self, f):

        return np.exp(f) if self.tranfcn == 'exp' else softplus(f)

    def df(self, y, f):

        if self.tranfcn == 'exp':
            return y - np.exp(f)
        else:
            return logistic(f) * (y / _safesoftplus(f) - 1)

    def d2f(self, y, f):

        if self.tranfcn == 'exp':
            return - np.exp(f)
        else:
            g = _safesoftplus(f)
            gp = logistic(f)
            g2p = gp * (1 - gp)
            return (y - g) * g2p / g - y * (gp / g)**2

    def d3f(self, y, f):

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

        return poisson.cdf(y, _safesoftplus(f))

    def interval(self, alpha, f):

        return poisson.interval(alpha, _safesoftplus(f))


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
