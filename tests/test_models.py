import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from revrand import StandardLinearModel, GeneralisedLinearModel
from revrand.likelihoods import Gaussian, Binomial
from revrand.basis_functions import LinearBasis, RandomRBF
from revrand.metrics import smse


def test_slm(make_gaus_data):

    X, y, w = make_gaus_data

    basis = LinearBasis(onescol=False)

    slm = StandardLinearModel(basis)
    slm.fit(X, y)
    Ey = slm.predict(X)

    assert smse(y, Ey) < 0.1

    basis = LinearBasis(onescol=False) + RandomRBF(nbases=10, Xdim=X.shape[1])

    slm = StandardLinearModel(basis)
    slm.fit(X, y)
    Ey = slm.predict(X)

    assert smse(y, Ey) < 0.1


def test_pipeline_slm(make_gaus_data):

    X, y, w = make_gaus_data

    slm = StandardLinearModel(LinearBasis(onescol=True))
    estimators = [('PCA', PCA()),
                  ('SLM', slm)]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1


def test_glm_gaussian(make_gaus_data):

    X, y, w = make_gaus_data

    basis = LinearBasis(onescol=True)
    lhood = Gaussian()

    # simple SGD
    glm = GeneralisedLinearModel(lhood, basis)
    glm.fit(X, y)
    Ey = glm.predict(X)
    assert smse(y, Ey) < 0.1

    # Test BasisCat
    basis = LinearBasis(onescol=True) + RandomRBF(nbases=10, Xdim=X.shape[1])

    glm = GeneralisedLinearModel(lhood, basis)
    glm.fit(X, y)
    Ey = glm.predict(X)
    assert smse(y, Ey) < 0.1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X)
    assert np.allclose(py, 1.)

    # Test log probability
    lpy, _, _ = glm.predict_logpdf(X, Ey)
    assert np.all(lpy > -100)

    EyQn, EyQx = glm.predict_interval(0.9, X)
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
    glm = GeneralisedLinearModel(lhood, basis)
    glm.fit(X, y, likelihood_args=largs)
    Ey = glm.predict(X, likelihood_args=largs)

    assert smse(f, Ey) < 1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(1e5, X, likelihood_args=largs)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(0.9, X, likelihood_args=largs)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)


def test_pipeline_glm(make_gaus_data):

    X, y, w = make_gaus_data

    glm = GeneralisedLinearModel(Gaussian(), LinearBasis(onescol=True))
    estimators = [('PCA', PCA()),
                  ('SLM', glm)
                  ]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1
