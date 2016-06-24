import pytest
import numpy as np
from scipy.stats import binom


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


@pytest.fixture
def make_gaus_data():

    w = np.array([1., 2.])
    x = np.atleast_2d(np.arange(-50, 50)).T
    X = np.hstack((np.ones((100, 1)), x))
    y = X.dot(w) + np.random.randn(100) / 1000

    return X, y, w


@pytest.fixture
def make_binom_data():

    X = np.atleast_2d(np.arange(-50, 50)).T

    p = 0.5 * (np.sin(X.flatten() / 5.) + 1)
    n = 1000
    y = binom.rvs(p=p, n=n)

    return X, y, p, n


@pytest.fixture
def make_cov():

    # Posdef
    X = np.random.randn(100, 5)
    S = np.cov(X.T)
    iS = np.linalg.pinv(S)

    # Slightly not posdef
    U, s, _ = np.linalg.svd(S)
    s[-1] = 0.
    Sn = (U * s).dot(U.T)
    iSn = np.linalg.pinv(Sn)

    return X, S, iS, Sn, iSn
