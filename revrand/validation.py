""" Some validation functions. """

from __future__ import division

import numpy as np


def smse(y_true, y_predict):
    """ Standardised mean squared error.

        Arguments:
            y_true: vector of true targest
            y_predict: vector of predicted targets

        Returns:
            SMSE of predictions vs truth (scalar)
    """

    N = y_true.shape[0]
    var = y_true.var()

    return ((y_true - y_predict)**2).sum() / (N * var)


def mll(y_true, y_predict, y_var):
    """ Mean log likelihood.

        TODO:
    """

    return normll(y_true, y_predict, y_var).mean()


def msll(y_true, y_predict, y_var, y_train):
    """ Mean standardised log likelihood.

        TODO:
    """

    var = y_train.var(ddof=1)
    mu = y_true.mean()
    logp_naive = -0.5 * (np.log(2 * np.pi * var) + (y_true - mu)**2 / var)
    return -(normll(y_true, y_predict, y_var) - logp_naive).mean()


def normll(y_true, y_predict, y_var):
    """ Gaussian log likelihood.

        TODO:
    """

    return -0.5 * (np.log(2 * np.pi * y_var) + (y_true - y_predict)**2 / y_var)
