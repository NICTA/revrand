import numpy as np
from scipy.stats import binom

from revrand import slm, glm
from revrand.likelihoods import Gaussian, Binomial
from revrand.basis_functions import LinearBasis, RandomRBF
from revrand.metrics import smse


def test_slm(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=False)

    params = slm.learn(X, y, basis)
    Ey, Vf, Vy = slm.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = slm.learn(X, y, basis)
    Ey, Vf, Vy = slm.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1


def test_glm_gaussian(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=True)
    lhood = Gaussian()

    # simple SGD
    params = glm.learn(X, y, lhood, basis)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    # simple LBFGS
    params = glm.learn(X, y, lhood, basis, use_sgd=False)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    # Test BasisCat
    basis = LinearBasis(onescol=True) + RandomRBF(nbases=10, Xdim=X.shape[1])

    # LBFGS
    params = glm.learn(X, y, lhood, basis, use_sgd=False)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    # SGD
    params = glm.learn(X, y, lhood, basis)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X, lhood, basis, *params)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(0.9, X, lhood, basis, *params)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)


def test_glm_binomial(make_data):
    # This is more to test the logic than to test if the model can overfit.
    # Because this is a somewhat pathalogical problem

    X, y, w = make_data
    n = int(y.max() + 1)
    yb = binom.rvs(p=(np.abs(y) / n), n=n, size=y.shape)
    n = (n,)

    basis = LinearBasis(onescol=True) + RandomRBF(nbases=20, Xdim=X.shape[1])
    lhood = Binomial()

    # SGD
    params = glm.learn(X, yb, lhood, basis, likelihood_args=n)
    Eyb, _, _, _ = glm.predict_moments(X, lhood, basis, *params,
                                       likelihood_args=n)
    assert smse(yb, Eyb) < smse(yb, np.ones_like(yb))

    # LBFGS
    params = glm.learn(X, yb, lhood, basis, use_sgd=False, likelihood_args=n)
    Eyb, _, _, _ = glm.predict_moments(X, lhood, basis, *params,
                                       likelihood_args=n)
    assert smse(yb, Eyb) < smse(yb, np.ones_like(yb))

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X, lhood, basis, *params,
                               likelihood_args=n)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(0.9, X, lhood, basis, *params,
                                      likelihood_args=n)
    assert all(Eyb <= EyQx)
    assert all(Eyb >= EyQn)
