import numpy as np

from revrand import slm, glm
from revrand.likelihoods import Gaussian, Binomial
from revrand.basis_functions import LinearBasis, RandomRBF
from revrand.metrics import smse


def test_slm(make_gaus_data):

    X, y, w = make_gaus_data

    basis = LinearBasis(onescol=False)

    params = slm.learn(X, y, basis)
    Ey, Vf, Vy = slm.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = slm.learn(X, y, basis)
    Ey, Vf, Vy = slm.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1


def test_glm_gaussian(make_gaus_data):

    X, y, w = make_gaus_data

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

    # Test log probability
    lpy, _, _ = glm.predict_logpdf(Ey, X, lhood, basis, *params)
    assert np.all(lpy > -100)

    EyQn, EyQx = glm.predict_interval(0.9, X, lhood, basis, *params)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)


def test_glm_binomial(make_binom_data):
    # This is more to test the logic than to test if the model can overfit,
    # hence more relaxed SMSE. This is because this is a harder problem than
    # the previous case.

    X, y, p, n = make_binom_data
    f = p * n

    basis = LinearBasis(onescol=True) + RandomRBF(nbases=20, Xdim=X.shape[1])
    lhood = Binomial()
    largs = (n,)

    # SGD
    params = glm.learn(X, y, lhood, basis, likelihood_args=largs, maxiter=3000,
                       batch_size=10)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params,
                                      likelihood_args=largs)

    assert smse(f, Ey) < 1

    # LBFGS
    params = glm.learn(X, y, lhood, basis, use_sgd=False, tol=1e-7,
                       maxiter=1000, likelihood_args=largs)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params,
                                      likelihood_args=largs)

    assert smse(f, Ey) < 1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X, lhood, basis, *params,
                               likelihood_args=largs)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(0.9, X, lhood, basis, *params,
                                      likelihood_args=largs)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)
