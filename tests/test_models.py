import numpy as np
from revrand import regression, glm
from revrand.likelihoods import Gaussian
from revrand.basis_functions import LinearBasis, RandomRBF
from revrand.validation import smse


def test_regression(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=False)

    params = regression.learn(X, y, basis)
    Ey, Vf, Vy = regression.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = regression.learn(X, y, basis)
    Ey, Vf, Vy = regression.predict(X, basis, *params)
    assert smse(y, Ey) < 0.1


def test_glm(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=False)
    lhood = Gaussian()

    params = glm.learn(X, y, lhood, basis)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = glm.learn(X, y, lhood, basis)
    Ey, _, _, _ = glm.predict_moments(X, lhood, basis, *params)
    assert smse(y, Ey) < 0.1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X, lhood, basis, *params)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(0.9, X, lhood, basis, *params)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)
