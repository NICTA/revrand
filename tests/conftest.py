import pytest
import numpy as np
from scipy.stats import binom

from sklearn.utils import check_random_state


# Test constants
RANDSTATE = 100
RANDOM = check_random_state(RANDSTATE)
NTRAIN = 400
NTEST = 200
NTOT = NTEST + NTRAIN


def split_data(X, y):

    trind = RANDOM.choice(NTOT, NTRAIN, replace=False)
    tsind = np.zeros(NTOT, dtype=bool)
    tsind[trind] = True
    tsind = np.where(~tsind)[0]

    return X[trind], y[trind], X[tsind], y[tsind]


# @pytest.fixture
# def make_randstate():
#     return RANDSTATE


@pytest.fixture
def make_random():
    return RANDOM


@pytest.fixture
def make_quadratic():

    # Find quadratic parameters y = a*x**2 + b*x + c
    a = 3.  # Keep these positive for the log-trick tests!!
    b = 2.
    c = 1.
    N = 1000

    bounds = [(None, None), (None, None), (1.1, None)]

    x = np.linspace(-1, 1, N)
    y = a * x**2 + b * x + c
    data = np.vstack((y, x)).T

    return a, b, c, data, bounds


# @pytest.fixture
# def make_gaus_data():

#     w = np.array([1., 2.])
#     x = np.linspace(-50, 50, NTOT)
#     X = np.hstack((np.ones((NTOT, 1)), x[:, np.newaxis]))
#     y = X.dot(w) + RANDOM.randn(NTOT) / 1000

#     return split_data(X, y)


@pytest.fixture
def make_gaus_data():

    x = np.linspace(-2, 2, NTOT)
    y = 3 + 2 * x
    X = np.hstack((np.ones((NTOT, 1)), x[:, np.newaxis]))

    return split_data(X, y)


@pytest.fixture
def make_binom_data():

    x = np.linspace(-50, 50, NTOT)
    X = np.atleast_2d(x).T

    p = 0.5 * (np.sin(x / 5.) + 1)
    n = 1000
    y = binom.rvs(p=p, n=n)

    return X, y, p, n


@pytest.fixture
def make_cov():

    # Posdef
    X = RANDOM.randn(100, 5)
    S = np.cov(X.T)
    iS = np.linalg.pinv(S)

    # Slightly not posdef
    l, U = np.linalg.eig(S)
    l[0] = -1e-13
    Sn = (U * l).dot(U.T)
    iSn = np.linalg.pinv(Sn)

    return X, S, iS, Sn, iSn
