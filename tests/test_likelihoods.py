from __future__ import division

import numpy as np
from scipy.stats import norm, poisson, bernoulli, binom

import revrand.likelihoods as lk

likelihoods = [lk.Gaussian, lk.Poisson, lk.Bernoulli, lk.Binomial]
likelihood_args = [[1.], [], [], [5]]


def test_shapes():

    N = 100
    y = np.ones(N)
    f = np.ones(N) * 2

    assert_shape = lambda x: x.shape == (N,)
    assert_args = lambda out, args: \
        all([o.shape == a.shape if not np.isscalar(a) else np.isscalar(o)
             for o, a in zip(out, args)])

    for like, args in zip(likelihoods, likelihood_args):

        lobj = like()
        assert_shape(lobj.loglike(y, f, *args))
        assert_shape(lobj.Ey(f, *args))
        assert_shape(lobj.df(y, f, *args))
        assert_shape(lobj.cdf(y, f, *args))
        assert_args(lobj.dp(y, f, *args), args)


def test_gaussian():

    # Test we can at match a Gaussian distribution from scipy

    mu = 0
    var = 2
    dist = lk.Gaussian()

    x = np.linspace(-10, 10, 100)

    p1 = norm.logpdf(x, loc=mu, scale=np.sqrt(var))
    p2 = dist.loglike(x, mu, var)

    np.allclose(p1, p2)

    p1 = norm.cdf(x, loc=mu, scale=np.sqrt(var))
    p2 = dist.cdf(x, mu, var)

    np.allclose(p1, p2)


def test_bernoulli():

    # Test we can at match a Bernoulli distribution from scipy

    p = 0.5
    dist = lk.Bernoulli()

    x = np.array([0, 1])

    p1 = bernoulli.logpmf(x, p)
    p2 = dist.loglike(x, p)

    np.allclose(p1, p2)

    p1 = bernoulli.cdf(x, p)
    p2 = dist.cdf(x, p)

    np.allclose(p1, p2)


def test_binom():

    # Test we can at match a Binomial distribution from scipy

    p = 0.5
    n = 5
    dist = lk.Binomial()

    x = np.random.randint(low=0, high=n, size=(10,))

    p1 = binom.logpmf(x, p=p, n=n)
    p2 = dist.loglike(x, p, n)

    np.allclose(p1, p2)

    p1 = binom.cdf(x, p=p, n=n)
    p2 = dist.cdf(x, p, n)

    np.allclose(p1, p2)


def test_poisson():

    # Test we can at match a Binomial distribution from scipy

    mu = 2
    dist = lk.Poisson()

    x = np.random.randint(low=0, high=5, size=(10,))

    p1 = poisson.logpmf(x, mu)
    p2 = dist.loglike(x, mu)

    np.allclose(p1, p2)

    p1 = poisson.cdf(x, mu)
    p2 = dist.cdf(x, mu)

    np.allclose(p1, p2)
