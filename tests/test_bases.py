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
