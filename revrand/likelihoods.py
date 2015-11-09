"""
Likelihood objects for inference within the GLM framework.
"""

import numpy as np
from scipy.special import gammaln
from .utils import Positive
from .transforms import logistic


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

        sig = logistic(f)
        return y * np.log(sig) + (1 - y) * np.log(1 - sig)

    def Ey(self, f):

        return logistic(f)

    def df(self, y, f):

        return y - logistic(f)

    def d2f(self, y, f):

        sig = logistic(f)
        return (sig - 1) * sig


class Gaussian(Bernoulli):

    def __init__(self, var_bounds=Positive()):

        self.bounds = [var_bounds]

    def loglike(self, y, f, var):

        return -0.5 * (np.log(2 * np.pi * var) + (y - f)**2 / var)

    def Ey(self, f, var):

        return f

    def df(self, y, f, var):

        return (y - f) / var

    def d2f(self, y, f, var):

        return - 1. / var * np.ones_like(f)


class Poisson(Bernoulli):

    def loglike(self, y, f):

        return y * f - gammaln(y + 1) - np.exp(f)

    def Ey(self, f):

        return np.exp(f)

    def df(self, y, f):

        return y - np.exp(f)

    def d2f(self, y, f):

        return - np.exp(f)
