""" Test the scikit learn pipeline interface. """

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from revrand.skl import StandardLinearModel, GeneralisedLinearModel
from revrand.basis_functions import LinearBasis
from revrand.likelihoods import Gaussian
from revrand.metrics import smse


def test_pipeline_slm(make_data):

    X, y, w = make_data

    estimators = [('PCA', PCA()),
                  ('SLM', StandardLinearModel(LinearBasis(onescol=True)))]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1

    Ey, _, Vy = pipe.predict_proba(X)
    assert smse(y, Ey) < 0.1


def test_pipeline_glm(make_data):

    X, y, w = make_data

    estimators = [('PCA', PCA()),
                  ('SLM', GeneralisedLinearModel(Gaussian(),
                                                 LinearBasis(onescol=True)))]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1

    Ey, ql, qu = pipe.predict_proba(X)
    assert smse(y, Ey) < 0.1
    assert all(ql < qu)
