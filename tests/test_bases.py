from __future__ import division

import numpy as np

import revrand.basis_functions as bs


def test_simple_concat(make_data):

    X1, X2 = make_data
    N, d1 = X1.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)
    P = base(X1)

    assert np.allclose(P, np.hstack((X1, X1)))

    base += bs.RadialBasis(centres=X1)
    P = base(X1, 1.)

    assert P.shape == (N, d1 * 2 + N)

    D = 200
    base += bs.RandomRBF_ARD(nbases=D, Xdim=d1)
    P = base(X1, 1., np.ones(d1))

    assert P.shape == (N, (D + d1) * 2 + N)


def test_grad_concat(make_data):

    X1, X2 = make_data
    N, d1 = X1.shape

    base = bs.LinearBasis(onescol=False) + bs.LinearBasis(onescol=False)

    assert list(base.grad(X1)) == []

    base += bs.RadialBasis(centres=X1)

    G = base.grad(X1, 1.)

    assert list(G)[0].shape == (N, N)

    D = 200
    base += bs.RandomRBF_ARD(nbases=D, Xdim=d1)
    G = base.grad(X1, 1., np.ones(d1))
    dims = [(N, N + D * 2), (N, N + D * 2, d1)]

    for g, d in zip(G, dims):
        assert g.shape == d


def test_apply_grad(make_data):

    X1, X2 = make_data
    N, d1 = X1.shape

    y = np.random.randn(N)

    def fun(Phi, dPhi):
        return y.dot(Phi).dot(dPhi.T).dot(y)

    base = bs.LinearBasis(onescol=False)
    obj = lambda dPhi: fun(base(X1), dPhi)

    assert bs.apply_grad(obj, base.grad(X1)) is None

    base = bs.RadialBasis(centres=X1)
    obj = lambda dPhi: fun(base(X1, 1.), dPhi)

    assert np.isscalar(bs.apply_grad(obj, base.grad(X1, 1.)))

    D = 200
    base = bs.RandomRBF_ARD(nbases=D, Xdim=d1)
    obj = lambda dPhi: fun(base(X1, np.ones(d1)), dPhi)

    assert bs.apply_grad(obj, base.grad(X1, np.ones(d1))).shape == (d1,)
