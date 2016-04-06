from __future__ import division

import numpy as np

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
