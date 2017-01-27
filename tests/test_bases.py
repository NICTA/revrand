from __future__ import division

import numpy as np
from operator import add
from functools import reduce
from scipy.stats import gamma

import revrand.basis_functions as bs
from revrand.btypes import Parameter, Positive, Bound
from revrand.utils import issequence


def test_simple_concat(make_gaus_data):

    X, _, _, _ = make_gaus_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)
    P = base.transform(X)

    assert np.allclose(P, np.hstack((X, X)))

    base += bs.RadialBasis(centres=X)
    P = base.transform(X, 1.)

    assert P.shape == (N, d * 2 + N)

    D = 200
    base += bs.RandomRBF(nbases=D, Xdim=d,
                         lenscale=Parameter(np.ones(d), Positive()))
    P = base.transform(X, 1., np.ones(d))

    assert P.shape == (N, (D + d) * 2 + N)


def test_concat_params():

    d = 10
    D = 20

    base = bs.LinearBasis(onescol=True) + bs.RandomMatern52(
        nbases=D,
        Xdim=d,
        lenscale=Parameter(1., Positive())
    )

    assert np.isscalar(base.params.value)

    base = bs.LinearBasis(onescol=True) + bs.RandomMatern52(
        nbases=D,
        Xdim=d,
        lenscale=Parameter(np.ones(d), Positive())
    )

    assert len(base.params.value) == d

    base += bs.RandomMatern52(
        nbases=D,
        Xdim=d,
        lenscale=Parameter(1., Positive())
    )

    assert len(base.params) == 2


def test_grad_concat(make_gaus_data):

    X, _, _, _ = make_gaus_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)

    assert list(base.grad(X)) == []

    base += bs.RadialBasis(centres=X)

    G = base.grad(X, 1.)

    assert list(G)[0].shape == (N, N + 2 * d)

    D = 200
    base += bs.RandomRBF(nbases=D, Xdim=d,
                         lenscale=Parameter(gamma(1), Positive(), shape=(d,)))
    G = base.grad(X, 1., np.ones(d))
    dims = [(N, N + (D + d) * 2), (N, N + (D + d) * 2, d)]

    for g, d in zip(G, dims):
        assert g.shape == d


def test_apply_grad(make_gaus_data):

    X, _, _, _ = make_gaus_data
    N, d = X.shape

    y = np.random.randn(N)

    def fun(Phi, dPhi):
        return y.dot(Phi).dot(dPhi.T).dot(y)

    base = bs.LinearBasis(onescol=False)
    obj = lambda dPhi: fun(base(X), dPhi)

    assert len(bs.apply_grad(obj, base.grad(X))) == 0

    base = bs.RadialBasis(centres=X)
    obj = lambda dPhi: fun(base.transform(X, 1.), dPhi)

    assert np.isscalar(bs.apply_grad(obj, base.grad(X, 1.)))

    D = 200
    base = bs.RandomRBF(nbases=D, Xdim=d,
                        lenscale=Parameter(np.ones(d), Positive()))
    obj = lambda dPhi: fun(base.transform(X, np.ones(d)), dPhi)

    assert bs.apply_grad(obj, base.grad(X, np.ones(d))).shape == (d,)

    base = bs.LinearBasis(onescol=False) + bs.RadialBasis(centres=X) \
        + bs.RandomRBF(nbases=D, Xdim=d,
                       lenscale=Parameter(np.ones(d), Positive()))
    obj = lambda dPhi: fun(base.transform(X, 1., np.ones(d)), dPhi)

    gs = bs.apply_grad(obj, base.grad(X, 1., np.ones(d)))
    assert np.isscalar(gs[0])
    assert gs[1].shape == (d,)


def test_bases(make_gaus_data, realhypers):

    X, _, _, _ = make_gaus_data
    N, d = X.shape
    nC = 10

    bases = [bs.BiasBasis(),
             bs.LinearBasis(onescol=True),
             bs.PolynomialBasis(order=2),
             bs.RadialBasis(centres=X[:nC, :]),
             bs.RadialBasis(centres=X[:nC, :],
                            lenscale=Parameter(np.ones(d), Positive())),
             bs.SigmoidalBasis(centres=X[:nC, :]),
             bs.SigmoidalBasis(centres=X[:nC, :],
                               lenscale=Parameter(np.ones(d), Positive())),
             bs.RandomRBF(Xdim=d, nbases=10),
             bs.RandomRBF(Xdim=d, nbases=10,
                          lenscale=Parameter(np.ones(d), Positive())),
             bs.OrthogonalRBF(Xdim=d, nbases=10),
             bs.OrthogonalRBF(Xdim=d, nbases=10,
                              lenscale=Parameter(np.ones(d), Positive())),
             bs.FastFoodRBF(Xdim=d, nbases=10),
             bs.FastFoodRBF(Xdim=d, nbases=10,
                            lenscale=Parameter(np.ones(d), Positive())),
             bs.FastFoodGM(Xdim=d, nbases=10),
             bs.FastFoodGM(Xdim=d, nbases=10,
                           mean=Parameter(np.zeros(d), Bound()),
                           lenscale=Parameter(np.ones(d), Positive())),
             ]

    if realhypers:
        hypers = [(),
                  (),
                  (),
                  (1.,),
                  (np.ones(d),),
                  (1.,),
                  (np.ones(d),),
                  (1.,),
                  (np.ones(d),),
                  (1.,),
                  (np.ones(d),),
                  (1.,),
                  (np.ones(d),),
                  (np.ones(d), np.ones(d)),
                  (np.ones(d), np.ones(d))
                  ]
    else:
        hypers = [(),
                  (),
                  (),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None,),
                  (None, None),
                  (None, None),
                  ]

    for b, h in zip(bases, hypers):
        P = b.transform(X, *h)
        dP = b.grad(X, *h)

        assert P.shape[0] == N
        if not issequence(dP):
            assert dP.shape[0] == N if not isinstance(dP, list) else dP == []
        else:
            for dp in dP:
                assert dp.shape[0] == N
        assert P.ndim == 2

    bcat = reduce(add, bases)
    hyps = []
    for h in hypers:
        hyps.extend(list(h))
    P = bcat.transform(X, *hyps)
    dP = bcat.grad(X, *hyps)

    assert bcat.get_dim(X) == P.shape[1]
    assert P.shape[0] == N
    assert P.ndim == 2
    for dp in dP:
        if not issequence(dP):
            assert dP.shape[0] == N if not isinstance(dP, list) else dP == []
        else:
            for dp in dP:
                assert dp.shape[0] == N


def test_slicing(make_gaus_data):

    X, _, _, _ = make_gaus_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False, apply_ind=[0]) \
        + bs.RandomRBF(Xdim=1, nbases=1, apply_ind=[1]) \
        + bs.RandomRBF(Xdim=2, nbases=3,
                       lenscale=Parameter(np.ones(2), Positive()),
                       apply_ind=[1, 0])

    P = base.transform(X, 1., np.ones(d))
    assert P.shape == (N, 9)

    dP = base.grad(X, 1., np.ones(d))
    assert list(dP)[0].shape == (N, 9)


def test_regularizer(make_gaus_data):

    X, _, _, _ = make_gaus_data
    N, d = X.shape
    nbases1, nbases2 = 10, 5

    # Single basis
    base = bs.LinearBasis(regularizer=Parameter(2, Positive()))
    diag, slices = base.regularizer_diagonal(X)
    assert base.regularizer.value == 2
    assert all(diag == np.full(d + 1, 2.))
    assert slices == slice(None)

    # Basis cat
    base += bs.RandomRBF(Xdim=d, nbases=nbases1) \
        + bs.RandomMatern32(Xdim=d, nbases=nbases2)
    dims = np.cumsum([0, d + 1, 2 * nbases1, 2 * nbases2])
    diag, slices = base.regularizer_diagonal(X)
    for db, de, s in zip(dims[:-1], dims[1:], slices):
        assert s == slice(db, de)
