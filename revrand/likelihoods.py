"""
Likelihood objects for inference within the GLM framework.
"""

import numpy as np
from .utils import Positive
from .transforms import logistic


class Gaussian():

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


class Bernoulli():

    def __init__(self):

        self.bounds = []

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
