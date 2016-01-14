from __future__ import division

import numpy as np

from revrand.optimize import sgd, minimize, struct_minimizer, \
    logtrick_minimizer
from revrand.utils import CatParameters, Bound, Positive, flatten


def test_unbounded(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = np.random.randn(3)

    res = sgd(qobj, w0, data, eval_obj=True, gtol=1e-4, passes=1000, rate=0.95,
              eta=1e-7)
    Ea, Eb, Ec = res['x']
    assert np.allclose((a, b, c), (Ea, Eb, Ec), atol=1e-2, rtol=0)

    res = minimize(qobj, w0, args=(data,), jac=True, method='L-BFGS-B')
    Ea, Eb, Ec = res['x']
    assert np.allclose((a, b, c), (Ea, Eb, Ec), atol=1e-3, rtol=0)

    res = minimize(qobj, w0, args=(data, False), jac=False, method=None,
                   xtol=1e-8)
    Ea, Eb, Ec = res['x']
    assert np.allclose((a, b, c), (Ea, Eb, Ec), atol=1e-3, rtol=0)


def test_bounded(make_quadratic):

    a, b, c, data, bounds = make_quadratic
    w0 = np.concatenate((np.random.randn(2), [1.5]))

    res = minimize(qobj, w0, args=(data,), jac=True, bounds=bounds,
                   method='L-BFGS-B')
    Ea_bfgs, Eb_bfgs, Ec_bfgs = res['x']

    res = sgd(qobj, w0, data, bounds=bounds, eval_obj=True, gtol=1e-4,
              passes=1000, rate=0.95, eta=1e-6)
    Ea_sgd, Eb_sgd, Ec_sgd = res['x']

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs),
                       (Ea_sgd, Eb_sgd, Ec_sgd),
                       atol=1e-2, rtol=0)


def test_structured_params(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = [np.random.randn(2), np.random.randn(1)[0]]

    nmin = struct_minimizer(minimize)
    res = nmin(qobj_struc, w0, args=(data,), jac=True, bounds=None,
               method='L-BFGS-B')
    (Ea_bfgs, Eb_bfgs), Ec_bfgs = res['x']

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs), (a, b, c), atol=1e-2,
                       rtol=0)


def test_log_params(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = np.abs(np.random.randn(3))
    bounds = [Positive(), Bound(), Positive()]

    nmin = logtrick_minimizer(minimize)
    res = nmin(qobj, w0, args=(data,), jac=True, bounds=bounds,
               method='L-BFGS-B')
    Ea_bfgs, Eb_bfgs, Ec_bfgs = res['x']

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs), (a, b, c), atol=1e-2,
                       rtol=0)


def test_logstruc_params(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = [np.abs(np.random.randn(2)), np.abs(np.random.randn(1))[0]]
    bounds = [Positive(), Bound(), Positive()]

    nmin = struct_minimizer(logtrick_minimizer(minimize))
    res = nmin(qobj_struc, w0, args=(data,), jac=True, bounds=bounds,
               method='L-BFGS-B')
    (Ea_bfgs, Eb_bfgs), Ec_bfgs = res['x']

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs), (a, b, c), atol=1e-2,
                       rtol=0)


def test_catparams(make_quadratic):

    a, b, c, data, _ = make_quadratic
    y, x = data[:, 0], data[:, 1]
    N = len(data)

    u = y - (a * x**2 + b * x + c)
    da, db, dc = -2 * np.array([(x**2 * u).sum(), (x * u).sum(), u.sum()]) / N

    params = [[a, b], c]
    grad = [[da, db], dc]

    # Test parameter flattening and logging
    pcat = CatParameters(params, log_indices=[1])
    fparams = pcat.flatten(params)
    assert all(np.array([a, b, np.log(c)]) == fparams)

    # Test gradient flattening and chain rule
    dparams = pcat.flatten_grads(params, grad)
    assert all(np.array([da, db, dc * c]) == dparams)

    # Test parameter reconstruction
    rparams = pcat.unflatten(fparams)
    for rp, p in zip(rparams, params):
        if not np.isscalar(rp):
            assert all(rp == p)
        else:
            assert rp == p


def qobj(w, data, grad=True):

    y, x = data[:, 0], data[:, 1]
    N = len(data)
    a, b, c = w

    u = y - (a * x**2 + b * x + c)
    f = (u**2).sum() / N
    df = -2 * np.array([(x**2 * u).sum(), (x * u).sum(), u.sum()]) / N

    if grad:
        return f, df
    else:
        return f


def qobj_struc(w12, w3, data, grad=True):

    return qobj(flatten([w12, w3])[0], data, grad)


if __name__ == "__main__":
    from conftest import make_quadratic
    test_unbounded(make_quadratic())
    test_bounded(make_quadratic())
    test_catparams(make_quadratic())
    test_structured_params(make_quadratic())
    test_log_params(make_quadratic())
    test_logstruc_params(make_quadratic())
