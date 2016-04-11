from revrand import regression, glm
from revrand.likelihoods import Gaussian
from revrand.basis_functions import LinearBasis, RandomRBF
from revrand.validation import smse


def test_regression(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=False)

    params = regression.learn(X, y, basis, [])
    Ey, Vf, Vy = regression.predict(X, basis, *params)

    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = regression.learn(X, y, basis, [1.])
    Ey, Vf, Vy = regression.predict(X, basis, *params)

    assert smse(y, Ey) < 0.1


def test_glm(make_data):

    X, y, w = make_data

    basis = LinearBasis(onescol=False)
    lhood = Gaussian()

    params = glm.learn(X, y, lhood, [1.], basis, [])
    Ey, _, _, _ = glm.predict_meanvar(X, lhood, basis, *params)

    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    params = glm.learn(X, y, lhood, [1.], basis, [1.])
    Ey, _, _, _ = glm.predict_meanvar(X, lhood, basis, *params)

    assert smse(y, Ey) < 0.1
