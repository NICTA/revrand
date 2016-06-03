from __future__ import division

import numpy as np

import revrand.math.special as tfms
import revrand.math.linalg as la


def test_logsumexp():

    X = np.random.randn(10, 3)

    # Axis 0
    lse_actual = np.log(np.sum(np.exp(X), axis=0))
    lse_test = tfms.logsumexp(X, axis=0)

    assert np.allclose(lse_actual, lse_test)

    # Axis 1
    lse_actual = np.log(np.sum(np.exp(X), axis=1))
    lse_test = tfms.logsumexp(X, axis=1)

    assert np.allclose(lse_actual, lse_test)


def test_softmax():

    X = np.random.randn(10, 3)

    # Axis 0
    sma0 = tfms.softmax(X, axis=0)
    assert np.allclose(sma0.sum(axis=0), np.ones(3))

    # Axis 1
    sma1 = tfms.softmax(X, axis=1)
    assert np.allclose(sma1.sum(axis=1), np.ones(10))


def test_sofplus():

    # Function validity test
    X = np.random.randn(10, 3)
    sp_actual = np.log(1 + np.exp(X))
    sp_test = tfms.softplus(X)

    assert np.allclose(sp_actual, sp_test)

    # Extreme value tests
    assert np.allclose(tfms.softplus(np.array([1e3])), 1e3 * np.ones(1))
    assert np.allclose(tfms.softplus(np.array([-1e3])), np.zeros(1))


def test_svd_solve(make_cov):

    X, S, iS, Sn, iSn = make_cov
    D = X.shape[1]

    U, s, _ = np.linalg.svd(S)
    iSpd, _ = la.svd_solve(U, s, np.eye(D))

    assert np.allclose(iS, iSpd)

    U, s, _ = np.linalg.svd(Sn)
    iSpd, _ = la.svd_solve(U, s, np.eye(D))

    assert np.allclose(iSn, iSpd)


def test_solve_posdef(make_cov):

    X, S, iS, Sn, iSn = make_cov
    D = X.shape[1]

    iSpd, _ = la.solve_posdef(S, np.eye(D))

    assert np.allclose(iS, iSpd)

    iSpd, _ = la.solve_posdef(Sn, np.eye(D))

    assert np.allclose(iSn, iSpd)
