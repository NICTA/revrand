import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone

from revrand import StandardLinearModel, GeneralisedLinearModel
from revrand.likelihoods import Gaussian, Binomial
from revrand.basis_functions import LinearBasis, RandomRBF, RandomMatern52
from revrand.metrics import smse


def test_slm(make_gaus_data):

    X, y, Xs, ys = make_gaus_data

    basis = LinearBasis(onescol=False)

    slm = StandardLinearModel(basis)
    slm.fit(X, y)
    Ey = slm.predict(Xs)

    assert smse(ys, Ey) < 0.1

    basis = LinearBasis(onescol=False) \
        + RandomRBF(nbases=10, Xdim=X.shape[1]) \
        + RandomMatern52(nbases=10, Xdim=X.shape[1])

    slm = StandardLinearModel(basis)
    slm.fit(X, y)
    Ey = slm.predict(Xs)

    assert smse(ys, Ey) < 0.1


def test_pipeline_slm(make_gaus_data):

    X, y, Xs, ys = make_gaus_data

    slm = StandardLinearModel(LinearBasis(onescol=True))
    estimators = [('PCA', PCA()),
                  ('SLM', slm)]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(Xs)
    assert smse(ys, Ey) < 0.1


def test_glm_gaussian(make_gaus_data, make_random):

    X, y, Xs, ys = make_gaus_data

    basis = LinearBasis(onescol=True)
    lhood = Gaussian()

    # simple SGD
    glm = GeneralisedLinearModel(lhood, basis, random_state=make_random)
    glm.fit(X, y)
    Ey = glm.predict(Xs)
    assert smse(ys, Ey) < 0.1

    # Test BasisCat
    basis = LinearBasis(onescol=True) \
        + RandomRBF(nbases=20, Xdim=X.shape[1]) \
        + RandomMatern52(nbases=20, Xdim=X.shape[1])

    glm = GeneralisedLinearModel(lhood, basis, random_state=make_random)
    glm.fit(X, y)
    Ey = glm.predict(Xs)
    assert smse(ys, Ey) < 0.1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(Xs, 1e5)
    assert np.allclose(py, 1.)

    # Test log probability
    lpy, _, _ = glm.predict_logpdf(Xs, Ey)
    assert np.all(lpy > -100)

    EyQn, EyQx = glm.predict_interval(Xs, 0.9)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)


def test_glm_binomial(make_binom_data, make_random):
    # This is more to test the logic than to test if the model can overfit,
    # hence more relaxed SMSE. This is because this is a harder problem than
    # the previous case. We also haven't split training ans test sets, since we
    # want to check the latent function and bounds

    X, y, p, n = make_binom_data
    f = p * n

    basis = LinearBasis(onescol=True) \
        + RandomRBF(nbases=20, Xdim=X.shape[1]) \
        + RandomMatern52(nbases=20, Xdim=X.shape[1])
    lhood = Binomial()
    largs = (n,)

    # SGD
    glm = GeneralisedLinearModel(lhood, basis, random_state=make_random)
    glm.fit(X, y, likelihood_args=largs)
    Ey = glm.predict(X, likelihood_args=largs)

    assert smse(f, Ey) < 1

    # Test upper quantile estimates
    py, _, _ = glm.predict_cdf(X, 1e5, likelihood_args=largs)
    assert np.allclose(py, 1.)

    EyQn, EyQx = glm.predict_interval(X, 0.9, likelihood_args=largs)
    assert all(Ey <= EyQx)
    assert all(Ey >= EyQn)


def test_pipeline_glm(make_gaus_data, make_random):

    X, y, Xs, ys = make_gaus_data

    glm = GeneralisedLinearModel(Gaussian(), LinearBasis(onescol=True),
                                 random_state=make_random)
    estimators = [('PCA', PCA()),
                  ('SLM', glm)
                  ]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(Xs)
    assert smse(ys, Ey) < 0.1


def test_sklearn_clone(make_gaus_data):

    X, y, Xs, ys = make_gaus_data

    basis = LinearBasis(onescol=True)
    slm = StandardLinearModel(basis=basis)
    glm = GeneralisedLinearModel(likelihood=Gaussian(), basis=basis,
                                 maxiter=100)

    slm_clone = clone(slm)
    glm_clone = clone(glm)

    slm_clone.fit(X, y)
    glm_clone.fit(X, y)

    # scalar values
    glm_keys = [
        'K',
        'batch_size',
        'maxiter',
        'nsamples',
        'random_state',
        'updater'
    ]

    for k in glm_keys:
        assert glm.get_params()[k] == glm_clone.get_params()[k]

    # Manually test likelihood and regulariser objects
    assert glm_clone.likelihood.params.value == glm.likelihood.params.value
    assert glm_clone.regulariser.value == glm.regulariser.value

    # scalar values
    slm_keys = [
        'maxiter',
        'tol'
    ]

    for k in slm_keys:
        assert slm.get_params()[k] == slm_clone.get_params()[k]

    # Manually test variance and regulariser objects
    assert slm_clone.var.value == slm.var.value
    assert slm_clone.regulariser.value == slm.regulariser.value
