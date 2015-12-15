from __future__ import division

import pytest
import nlopt
import numpy as np

from revrand.optimize.nlopt_wrap import make_nlopt_fun
from revrand.utils import couple

from scipy.optimize import rosen, rosen_der

rosen_couple = couple(rosen, rosen_der)


@pytest.fixture
def start_point():
    return [1.3, 0.7, 0.8, 1.9, 1.2]


def test_make_nlopt_fun0(start_point):
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=rosen_der)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad1(start_point):
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    obj_fun = make_nlopt_fun(rosen_couple, jac=rosen_der)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad2(start_point):
    # If a callable jacobian `jac` is specified, it will take precedence
    # over the gradient given by a function that returns a tuple with the
    # gradient as its second value.
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    # We give some function that is clearly not the correct derivative.
    obj_fun = make_nlopt_fun(couple(rosen, lambda x: 2 * x), jac=rosen_der)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad3(start_point):
    # If you use a gradient-based optimization method with `jac=True` but
    # fail to supply any gradient information, you will receive a
    # `RuntimeWarning` and poor results.
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=True)
    opt.set_min_objective(obj_fun)
    with pytest.warns(RuntimeWarning):
        x_opt = opt.optimize(x0)
    assert np.allclose(x_opt, x0)


def test_make_nlopt_fun_grad4(start_point):
    # Likewise, if you *do* supply gradient information, but set `jac=False`
    # you will be reminded of the fact that the gradient information is
    # being ignored through a `RuntimeWarning`.
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    obj_fun = make_nlopt_fun(rosen_couple, jac=False)
    opt.set_min_objective(obj_fun)
    with pytest.warns(RuntimeWarning):
        x_opt = opt.optimize(x0)
    assert np.allclose(x_opt, x0)


def test_make_nlopt_fun_grad5(start_point):
    # Of course, you can use gradient-based optimization and not supply
    # any gradient information at your own discretion.
    # No warning are raised.
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=False)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), x0)


def test_make_nlopt_fun_neldermead(start_point):
    x0 = start_point
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=False)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_bobyqa(start_point):
    x0 = start_point
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=False)
    opt.set_min_objective(obj_fun)
    opt.set_ftol_abs(1e-11)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad_free0(start_point):
    # When using derivative-free optimization methods, gradient information
    # supplied in any form is disregarded without warning.
    x0 = start_point
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    obj_fun = make_nlopt_fun(rosen, jac=rosen_der)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad_free1(start_point):
    # When using derivative-free optimization methods, gradient information
    # supplied in any form is disregarded without warning.
    x0 = start_point
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    obj_fun = make_nlopt_fun(rosen_couple, jac=True)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)


def test_make_nlopt_fun_grad_path(start_point):
    x0 = start_point
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    cache = []
    obj_fun = make_nlopt_fun(rosen, jac=rosen_der, xs=cache)
    opt.set_min_objective(obj_fun)
    assert np.allclose(opt.optimize(x0), np.array([1., 1., 1., 1., 1.]))
    assert np.isclose(opt.last_optimum_value(), 0)
    assert len(cache) == 51
    assert np.allclose(np.array(cache).round(2),
                       np.array(
        [[1.3, 0.7, 0.8, 1.9, 1.2],
         [-514.1, 286.1, 342.4, -2083.5, 483.2],
         [-170.83, 96.02, 114.89, -694.57, 162.18],
         [-56.41, 32.66, 39.05, -231.6, 55.17],
         [-18.27, 11.53, 13.77, -77.26, 19.5],
         [-5.55, 4.49, 5.34, -25.8, 7.6],
         [-1.29, 2.14, 2.52, -8.59, 3.63],
         [0.16, 1.33, 1.56, -2.71, 2.27],
         [0.78, 0.99, 1.15, -0.22, 1.69],
         [0.84, 0.99, 0.65, -0.18, 1.48],
         [0.94, 0.81, 0.45, -0.17, 1.16],
         [0.86, 0.66, 0.15, -0.14, 0.31],
         [0.68, 0.45, 0.11, -0.05, 0.04],
         [0.64, 0.4, 0.12, 0., -0.],
         [0.62, 0.39, 0.14, 0.03, -0.],
         [0.63, 0.39, 0.16, 0.03, 0.],
         [0.64, 0.41, 0.18, 0.04, 0.01],
         [0.69, 0.48, 0.25, 0.07, 0.02],
         [0.95, 0.83, 0.64, 0.2, 0.08],
         [0.79, 0.61, 0.4, 0.12, 0.04],
         [0.8, 0.64, 0.43, 0.15, 0.03],
         [0.88, 0.76, 0.53, 0.24, 0.02],
         [0.84, 0.7, 0.48, 0.2, 0.02],
         [0.8, 0.66, 0.42, 0.24, 0.],
         [0.83, 0.69, 0.47, 0.21, 0.02],
         [0.84, 0.7, 0.48, 0.22, 0.03],
         [0.89, 0.78, 0.57, 0.32, 0.09],
         [0.89, 0.8, 0.62, 0.38, 0.14],
         [0.95, 0.91, 0.81, 0.6, 0.28],
         [0.92, 0.84, 0.7, 0.47, 0.2],
         [0.92, 0.86, 0.74, 0.54, 0.28],
         [0.94, 0.89, 0.79, 0.61, 0.37],
         [0.97, 0.94, 0.88, 0.76, 0.54],
         [0.96, 0.92, 0.84, 0.69, 0.47],
         [0.97, 0.94, 0.88, 0.79, 0.63],
         [0.98, 0.96, 0.92, 0.84, 0.69],
         [0.97, 0.94, 0.89, 0.78, 0.62],
         [0.98, 0.95, 0.91, 0.82, 0.67],
         [0.98, 0.96, 0.92, 0.85, 0.71],
         [0.99, 0.98, 0.96, 0.92, 0.84],
         [0.99, 0.98, 0.97, 0.94, 0.88],
         [1., 1., 0.99, 0.98, 0.96],
         [1., 1., 0.99, 0.99, 0.97],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]]))
