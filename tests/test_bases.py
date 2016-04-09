from __future__ import division

import numpy as np
from operator import add
from functools import reduce

import revrand.basis_functions as bs


def test_simple_concat(make_data):

    X, _, _ = make_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)
    P = base(X)

    assert np.allclose(P, np.hstack((X, X)))

    base += bs.RadialBasis(centres=X)
    P = base(X, 1.)

    assert P.shape == (N, d * 2 + N)

    D = 200
    base += bs.RandomRBF_ARD(nbases=D, Xdim=d)
    P = base(X, 1., np.ones(d))

    assert P.shape == (N, (D + d) * 2 + N)


def test_grad_concat(make_data):

    X, _, _ = make_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)

    assert list(base.grad(X)) == []

    base += bs.RadialBasis(centres=X)

    G = base.grad(X, 1.)

    assert list(G)[0].shape == (N, N + 2 * d)

    D = 200
    base += bs.RandomRBF_ARD(nbases=D, Xdim=d)
    G = base.grad(X, 1., np.ones(d))
    dims = [(N, N + (D + d) * 2), (N, N + (D + d) * 2, d)]

    for g, d in zip(G, dims):
        assert g.shape == d


def test_apply_grad(make_data):

    X, _, _ = make_data
    N, d = X.shape

    y = np.random.randn(N)

    def fun(Phi, dPhi):
        return y.dot(Phi).dot(dPhi.T).dot(y)

    base = bs.LinearBasis(onescol=False)
    obj = lambda dPhi: fun(base(X), dPhi)

    assert len(bs.apply_grad(obj, base.grad(X))) == 0

    base = bs.RadialBasis(centres=X)
    obj = lambda dPhi: fun(base(X, 1.), dPhi)

    assert np.isscalar(bs.apply_grad(obj, base.grad(X, 1.)))

    D = 200
    base = bs.RandomRBF_ARD(nbases=D, Xdim=d)
    obj = lambda dPhi: fun(base(X, np.ones(d)), dPhi)

    assert bs.apply_grad(obj, base.grad(X, np.ones(d))).shape == (d,)

    base = bs.LinearBasis(onescol=False) + bs.RadialBasis(centres=X) \
        + bs.RandomRBF_ARD(nbases=D, Xdim=d)
    obj = lambda dPhi: fun(base(X, 1., np.ones(d)), dPhi)

    gs = bs.apply_grad(obj, base.grad(X, 1., np.ones(d)))
    assert np.isscalar(gs[0])
    assert gs[1].shape == (d,)


def test_bases(make_data):

    X, _, _ = make_data
    N, d = X.shape

    bases = [bs.LinearBasis(onescol=True),
             bs.PolynomialBasis(order=2),
             bs.RadialBasis(centres=X[:10, :]),
             bs.SigmoidalBasis(centres=X[:10, :]),
             bs.RandomRBF(Xdim=d, nbases=10),
             bs.RandomRBF_ARD(Xdim=d, nbases=10),
             bs.FastFood(Xdim=d, nbases=10),
             ]

    hypers = [None,
              None,
              1.,
              1.,
              1.,
              np.ones(d),
              1.
              ]

    for b, h in zip(bases, hypers):
        P = b(X, h) if h is not None else b(X)
        dP = b.grad(X, h) if h is not None else b.grad(X)

        assert P.shape[0] == N
        assert dP.shape[0] == N if not isinstance(dP, list) else dP == []
        assert P.ndim == 2

    bcat = reduce(add, bases)
    hyps = [h for h in hypers if h is not None]
    P = bcat(X, *hyps)
    dP = bcat.grad(X, *hyps)

    assert P.shape[0] == N
    assert P.ndim == 2
    for dp in dP:
        assert dp.shape[0] == N if not isinstance(dp, list) else dp == []


def test_slicing(make_data):

    X, _, _ = make_data
    N, d = X.shape

    base = bs.LinearBasis(onescol=False, apply_ind=[0]) \
        + bs.RandomRBF(Xdim=1, nbases=1, apply_ind=[1]) \
        + bs.RandomRBF_ARD(Xdim=d, nbases=3, apply_ind=[1, 0])

    P = base(X, 1., np.ones(d))
    assert P.shape == (N, 9)

    dP = base.grad(X, 1., np.ones(d))
    assert list(dP)[0].shape == (N, 9)
