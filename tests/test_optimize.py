from __future__ import division

import numpy as np
from scipy.optimize import minimize

from revrand.optimize import sgd, structured_minimizer, logtrick_minimizer, \
    structured_sgd, logtrick_sgd, AdaDelta, Momentum, AdaGrad, SGDUpdater
from revrand.btypes import Bound, Positive, Parameter
from revrand.utils import flatten


def test_unbounded(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = np.random.randn(3)

    assert_opt = lambda Ea, Eb, Ec: \
        np.allclose((a, b, c), (Ea, Eb, Ec), atol=1e-3, rtol=0)

    for updater in [SGDUpdater, AdaDelta, AdaGrad, Momentum]:
        res = sgd(qobj, w0, data, eval_obj=True, gtol=1e-4, passes=1000,
                  updater=updater())
        assert_opt(*res['x'])

    res = minimize(qobj, w0, args=(data,), jac=True, method='L-BFGS-B')
    assert_opt(*res['x'])

    res = minimize(qfun, w0, args=(data,), jac=qgrad, method='L-BFGS-B')
    assert_opt(*res['x'])

    res = minimize(qfun, w0, args=(data), jac=False, method=None)
    assert_opt(*res['x'])


def test_bounded(make_quadratic):

    a, b, c, data, bounds = make_quadratic
    w0 = np.concatenate((np.random.randn(2), [1.5]))

    res = minimize(qobj, w0, args=(data,), jac=True, bounds=bounds,
                   method='L-BFGS-B')
    Ea_bfgs, Eb_bfgs, Ec_bfgs = res['x']

    res = sgd(qobj, w0, data, bounds=bounds, eval_obj=True, gtol=1e-4,
              passes=1000)
    Ea_sgd, Eb_sgd, Ec_sgd = res['x']

    assert np.allclose((Ea_bfgs, Eb_bfgs, Ec_bfgs),
                       (Ea_sgd, Eb_sgd, Ec_sgd),
                       atol=1e-2, rtol=0)


def test_structured_params(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = [Parameter(np.random.randn(2), Bound()),
          Parameter(np.random.randn(1), Bound())
          ]

    qobj_struc = lambda w12, w3, data: q_struc(w12, w3, data, qobj)
    assert_opt = lambda Eab, Ec: \
        np.allclose((a, b, c), (Eab[0], Eab[1], Ec), atol=1e-3, rtol=0)

    nmin = structured_minimizer(minimize)
    res = nmin(qobj_struc, w0, args=(data,), jac=True, method='L-BFGS-B')
    assert_opt(*res.x)

    nsgd = structured_sgd(sgd)
    res = nsgd(qobj_struc, w0, data, eval_obj=True, gtol=1e-4, passes=1000)
    assert_opt(*res.x)

    qf_struc = lambda w12, w3, data: q_struc(w12, w3, data, qfun)
    qg_struc = lambda w12, w3, data: q_struc(w12, w3, data, qgrad)
    res = nmin(qf_struc, w0, args=(data,), jac=qg_struc, method='L-BFGS-B')
    assert_opt(*res.x)


def test_log_params(make_quadratic):

    a, b, c, data, _ = make_quadratic
    w0 = np.abs(np.random.randn(3))
    bounds = [Positive(), Bound(), Positive()]

    assert_opt = lambda Ea, Eb, Ec: \
        np.allclose((a, b, c), (Ea, Eb, Ec), atol=1e-3, rtol=0)

    nmin = logtrick_minimizer(minimize)
    res = nmin(qobj, w0, args=(data,), jac=True, method='L-BFGS-B',
               bounds=bounds)
    assert_opt(*res.x)

    nsgd = logtrick_sgd(sgd)
    res = nsgd(qobj, w0, data, eval_obj=True, gtol=1e-4, passes=1000,
               bounds=bounds)
    assert_opt(*res.x)

    nmin = logtrick_minimizer(minimize)
    res = nmin(qfun, w0, args=(data,), jac=qgrad, method='L-BFGS-B',
               bounds=bounds)
    assert_opt(*res.x)


def test_logstruc_params(make_quadratic):

    a, b, c, data, _ = make_quadratic

    w0 = [Parameter(np.random.randn(2), Positive()),
          Parameter(np.random.randn(1), Bound())
          ]

    qobj_struc = lambda w12, w3, data: q_struc(w12, w3, data, qobj)
    assert_opt = lambda Eab, Ec: \
        np.allclose((a, b, c), (Eab[0], Eab[1], Ec), atol=1e-3, rtol=0)

    nmin = structured_minimizer(logtrick_minimizer(minimize))
    res = nmin(qobj_struc, w0, args=(data,), jac=True, method='L-BFGS-B')
    assert_opt(*res.x)

    nsgd = structured_sgd(logtrick_sgd(sgd))
    res = nsgd(qobj_struc, w0, data, eval_obj=True, gtol=1e-4, passes=1000)
    assert_opt(*res.x)

    qf_struc = lambda w12, w3, data: q_struc(w12, w3, data, qfun)
    qg_struc = lambda w12, w3, data: q_struc(w12, w3, data, qgrad)
    res = nmin(qf_struc, w0, args=(data,), jac=qg_struc, method='L-BFGS-B')
    assert_opt(*res.x)


def qfun(w, data):

    x, u = _getxu(w, data)
    N = len(x)

    f = (u**2).sum() / N
    return f


def qgrad(w, data):

    x, u = _getxu(w, data)
    N = len(x)

    df = -2 * np.array([(x**2 * u).sum(), (x * u).sum(), u.sum()]) / N
    return df


def qobj(w, data):

    return qfun(w, data), qgrad(w, data)


def _getxu(w, data):

    y, x = data[:, 0], data[:, 1]
    a, b, c = w
    u = y - (a * x**2 + b * x + c)

    return x, u


def q_struc(w12, w3, data, func):

    return func(flatten([w12, w3], returns_shapes=False), data)
