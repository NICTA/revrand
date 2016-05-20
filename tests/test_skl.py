""" Test the scikit learn pipeline interface. """

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import revrand.skl as skl
import revrand.basis_functions as bf
from revrand.likelihoods import Gaussian
from revrand.metrics import smse


def test_pipeline_slm(make_data):

    X, y, w = make_data

    slm = skl.StandardLinearModel(bf.LinearBasis(onescol=True))
    estimators = [('PCA', PCA()),
                  ('SLM', slm)]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1

    Ey, _, Vy = pipe.predict_proba(X)
    assert smse(y, Ey) < 0.1


def test_pipeline_glm(make_data):

    X, y, w = make_data

    glm = skl.GeneralisedLinearModel(Gaussian(), bf.LinearBasis(onescol=True))
    estimators = [('PCA', PCA()),
                  ('SLM', glm)
                  ]
    pipe = Pipeline(estimators)

    pipe.fit(X, y)
    Ey = pipe.predict(X)
    assert smse(y, Ey) < 0.1

    Ey, ql, qu = pipe.predict_proba(X)
    assert smse(y, Ey) < 0.1
    assert all(ql < qu)


def test_pipeline_bases(make_data):

    X, y, w = make_data

    exactbases = [(bf.LinearBasis,
                   skl.LinearBasis,
                   {'onescol': True},
                   {}
                   )
                  ]

    randombases = [(bf.RandomRBF,
                    skl.RandomRBF,
                    {'nbases': 20, 'Xdim': X.shape[1]},
                    {'lenscale': 1.}
                    )
                   ]

    for bfbase, sklbase, inits, params in exactbases:

        estimators = [('base', sklbase(**dict_join(inits, params)))]

        pipe = Pipeline(estimators)
        pipe.fit(X)  # Should do nothing really

        assert np.allclose(pipe.transform(X), bfbase(**inits)(X, **params))

    for bfbase, sklbase, inits, params in randombases:

        estimators = [('base', sklbase(**dict_join(inits, params)))]

        pipe = Pipeline(estimators)
        pipe.fit(X)  # Should do nothing really

        assert pipe.transform(X).shape == bfbase(**inits)(X, **params).shape

    # Test Pickling
    # for _, sklbase, inits, params in (exactbases + randombases):

    #     estimators = [('base', sklbase(**dict_join(inits, params)))]

    #     pipe = Pipeline(estimators)
    #     pipe.fit(X)  # Should do nothing really


def dict_join(*dicts):

    rdict = dicts[0].copy()
    for d in dicts[1:]:
        rdict.update(d)

    return rdict
