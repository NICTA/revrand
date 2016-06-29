from __future__ import division

import numpy as np
from scipy.stats import beta

import revrand.likelihoods as lk

likelihoods = [lk.Gaussian, lk.Poisson, lk.Bernoulli, lk.Binomial, lk.Beta3]
likelihood_args = [[1.], [], [], [5], [2, 5]]


def test_shapes():

    N = 100
    y = np.ones(N)
    f = np.ones(N) * 2

    assert_shape = lambda x: x.shape == (N,)
    assert_args = lambda out, args: \
        all([o.shape == a.shape if not np.isscalar(a) else np.isscalar(o)
             for o, a in zip(out, args)])
    assert_ashape = lambda x: \
        all([xi.shape == (N,) for xi in x]) if not np.isscalar(x) \
        else x.shape == (N,)

    for like, args in zip(likelihoods, likelihood_args):

        lobj = like()
        assert_shape(lobj.loglike(y, f, *args))
        assert_shape(lobj.Ey(f, *args))
        assert_shape(lobj.df(y, f, *args))
        assert_shape(lobj.d2f(y, f, *args))
        assert_shape(lobj.d3f(y, f, *args))
        assert_shape(lobj.cdf(y, f, *args))
        assert_args(lobj.dp(y, f, *args), args)
        assert_ashape(lobj.dpd2f(y, f, *args))


def test_beta3_loglike():

    x = np.linspace(0.001, 1 - 0.001, 100)
    a = 2.
    b = 5.
    f = 1.

    ll_beta = beta.logpdf(x, a, b)
    ll_beta3 = lk.Beta3().loglike(x, a, b, f)

    np.allclose(ll_beta, ll_beta3)
