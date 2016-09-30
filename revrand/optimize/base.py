"""Optimize Base Decorators."""
import numpy as np

from itertools import repeat
from six import wraps
from sklearn.utils import check_random_state

import revrand.btypes as bt
from ..utils import flatten, unflatten


# Constants
MINPOS = 1e-100  # Min for log trick warped data
MAXPOS = np.sqrt(np.finfo(float).max)  # Max for log trick warped data
LOGMINPOS = np.log(MINPOS)
EXPMAX = np.log(MAXPOS)


def candidate_start_points_random(bounds, n_candidates=1000,
                                  random_state=None):
    r"""
    Randomly generate starting points uniformly within a hyperrectangle.

    Parameters
    ----------
    bounds : list of tuples (pairs)
        List of one or more bound pairs

    n_candidates : int
        Number of candidate starting points to generate

    Returns
    -------
    ndarray
        Array of shape (len(bounds), n_candidates)

    Notes
    -----
    Roughly equivalent to::

        lambda bounds, n_candidates=100: \
            np.random.uniform(*zip(*bounds), size=(n_candidates,len(bounds))).T

    Examples
    --------
    >>> candidate_start_points_random([(-10., -3.5), (-1., 2.)],
    ...     n_candidates=5, random_state=1)
    array([[-7.28935697, -9.99925656, -9.04608671, -8.78930863, -7.42101142],
           [ 1.16097348, -0.09300228, -0.72298422,  0.03668218,  0.6164502 ]])

    >>> candidate_start_points = candidate_start_points_random(
    ...     [(-10., -3.5), (-1., 2.)], random_state=1)

    >>> candidate_start_points.shape
    (2, 1000)

    >>> np.all(-10 <= candidate_start_points[0])
    True
    >>> np.all(candidate_start_points[0] < -3.5)
    True

    >>> np.all(-1. < candidate_start_points[1])
    True
    >>> np.all(candidate_start_points[1] <= 2.)
    True

    Uniformly sample from line segment:

    >>> candidate_start_points_random([(-1., 2.)], n_candidates=5,
    ...                               random_state=1)
    array([[ 0.25106601,  1.16097348, -0.99965688, -0.09300228, -0.55973233]])

    Uniformly sample from hyperrectangle:

    >>> candidate_start_points_random([(-10., -3.5), (-1., 2.), (5., 7.),
    ... (2.71, 3.14)], n_candidates=5, random_state=1)
    array([[-7.28935697, -9.04608671, -7.42101142, -8.67106038, -7.28751878],
           [ 1.16097348, -0.72298422,  0.6164502 ,  1.63435231,  0.67606949],
           [ 5.00022875,  5.37252042,  5.83838903,  5.05477519,  5.28077388],
           [ 2.84000301,  2.85859111,  3.00464439,  2.99830103,  2.79518364]])
    """
    generator = check_random_state(random_state)

    low, high = zip(*bounds)
    n_dims = len(bounds)
    return generator.uniform(low, high, (n_candidates, n_dims)).transpose()


def candidate_start_points_lattice(bounds, nums=3):
    r"""
    Generate starting points on a uniform grid within a hyperrectangle.

    Parameters
    ----------
    bounds : list of tuples (pairs)
        List of one or more bound pairs

    nums : int (optional)
        number of grid points per dimension

    Returns
    -------
    ndarray
        Array of shape (len(bounds), prod(nums))

    Examples
    --------
    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)],
    ...                                nums=[5, 3]).shape
    (2, 15)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                                nums=[5, 10, 9]) # doctest: +ELLIPSIS
    array([[-1.   , -1.   , -1.   , ...,  1.5  ,  1.5  ,  1.5  ],
           [-1.5  , -1.5  , -1.5  , ...,  3.   ,  3.   ,  3.   ],
           [ 0.   ,  0.625,  1.25 , ...,  3.75 ,  4.375,  5.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                             nums=[5, 10, 9]).shape
    (3, 450)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5), (1, 5)],
    ...                                nums=[5, 10, 9, 3]) # doctest: +ELLIPSIS
    array([[-1. , -1. , -1. , ...,  1.5,  1.5,  1.5],
           [-1.5, -1.5, -1.5, ...,  3. ,  3. ,  3. ],
           [ 0. ,  0. ,  0. , ...,  5. ,  5. ,  5. ],
           [ 1. ,  3. ,  5. , ...,  1. ,  3. ,  5. ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5), (1, 5)],
    ...                             nums=[5, 10, 9, 3]).shape
    (4, 1350)

    Third ``num`` is ignored

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=[5, 3, 9])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    Third bound is ignored

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                                nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)]).shape
    (2, 9)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=9).shape
    (2, 81)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                                nums=2).shape
    (3, 8)
    """
    if isinstance(nums, int):
        nums = repeat(nums)

    linspaces = [np.linspace(start, end, num) for (start, end), num
                 in zip(bounds, nums)]
    return np.vstack(a.flatten() for a in np.meshgrid(*linspaces))


def minimize_bounded_start(candidates_func=candidate_start_points_random,
                           *candidates_func_args, **candidates_func_kwargs):
    """
    Decorator for selecting the best optimiser starting point.

    The starting point lies within a hyperrectangle, and all points are tested
    in :code:`candidates_func`.

    See Also
    --------
    candidate_start_points_random, candidate_start_points_lattice

    Examples
    --------
    >>> from scipy.optimize import minimize as sp_min, rosen, rosen_der

    >>> @minimize_bounded_start(n_candidates=250, random_state=1)
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)

    >>> rect = [(-1, 1.5), (-.5, 1.5)]

    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> np.allclose(res.x, np.array([ 1.,  1.]))
    True

    >>> np.isclose(res.fun, 0)
    True

    >>> res.start.round(2)
    array([ 1.17,  1.4 ])

    There are several other ways to use this decorator:

    >>> @minimize_bounded_start()
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)

    >>> minimize_bounded_start_dec = minimize_bounded_start(n_candidates=250)

    >>> @minimize_bounded_start_dec
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)

    >>> my_min = minimize_bounded_start_dec(sp_min)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)

    >>> @minimize_bounded_start(candidate_start_points_lattice, nums=[5, 9])
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> res.start
    array([ 0.875,  0.75 ])

    >>> minimize_bounded_start_dec = minimize_bounded_start(
    ...     candidate_start_points_lattice, nums=[5, 9])

    >>> my_min = minimize_bounded_start_dec(sp_min)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> res.start
    array([ 0.875,  0.75 ])

    Just to confirm this is the correct starting point:

    >>> candidates = candidate_start_points_lattice(rect, nums=[5, 9])
    >>> candidates[:, rosen(candidates).argmin()]
    array([ 0.875,  0.75 ])
    """
    def minimize_bounded_start_dec(minimize_func):

        @wraps(minimize_func)
        def _minimize_bounded_start(fun, x0_bounds, *args, **kwargs):
            candidate_start_points = candidates_func(x0_bounds,
                                                     *candidates_func_args,
                                                     **candidates_func_kwargs)
            candidate_start_values = fun(candidate_start_points)
            min_start_point_ind = np.argmin(candidate_start_values)
            min_start_point = candidate_start_points[:, min_start_point_ind]
            res = minimize_func(fun, min_start_point, *args, **kwargs)
            res.start = min_start_point
            return res

        return _minimize_bounded_start

    return minimize_bounded_start_dec


def structured_minimizer(minimizer):
    """
    Allow an optimizer to accept a list of Parameter types to optimize.

    Examples
    --------
    >>> from scipy.optimize import minimize as sp_min
    >>> from ..btypes import Parameter, Bound

    Define a cost function that returns a pair. The first element is the cost
    value and the second element is the gradient represented by a tuple. Even
    if the cost is a function of a single variable, the gradient must be a
    tuple containing one element.

    >>> def cost(w, lambda_):
    ...     sq_norm = w.T.dot(w)
    ...     return .5 * lambda_ * sq_norm, (lambda_ * w, .5 * sq_norm)

    Augment the Scipy optimizer to take structured inputs

    >>> new_min = structured_minimizer(sp_min)

    Initial values

    >>> w_0 = Parameter(np.array([.5, .1, .2]), Bound())
    >>> lambda_0 = Parameter(.25, Bound())

    >>> res = new_min(cost, (w_0, lambda_0), method='L-BFGS-B', jac=True)
    >>> res_w, res_lambda = res.x
    """
    @wraps(minimizer)
    def new_minimizer(fun, parameters, jac=True, **minimizer_kwargs):

        (array1d, fbounds), shapes = flatten(parameters,
                                             hstack=bt.hstack,
                                             shape=bt.shape,
                                             ravel=bt.ravel
                                             )
        flatten_args_dec = flatten_args(shapes)

        new_fun = flatten_args_dec(fun)

        if callable(jac):
            new_jac = flatten_args_dec(jac)
        else:
            new_jac = jac
            if bool(jac):
                new_fun = flatten_func_grad(new_fun)

        result = minimizer(new_fun, array1d, jac=new_jac, bounds=fbounds,
                           **minimizer_kwargs)
        result['x'] = tuple(unflatten(result['x'], shapes))

        if bool(jac):
            result['jac'] = tuple(unflatten(result['jac'], shapes))

        return result

    return new_minimizer


def structured_sgd(sgd):
    """
    Allow stochastic gradients to accept a list of Parameter types to optimize.

    Examples
    --------
    >>> from ..optimize import sgd
    >>> from ..btypes import Parameter, Bound

    Define a cost function that returns a pair. The first element is the cost
    value and the second element is the gradient represented by a sequence.
    Even if the cost is a function of a single variable, the gradient must be a
    sequence containing one element.

    >>> def cost(w, lambda_, data):
    ...     N = len(data)
    ...     y, X = data[:, 0], data[:, 1:]
    ...     y_est = X.dot(w)
    ...     ww = w.T.dot(w)
    ...     obj = (y - y_est).sum() / N + lambda_ * ww
    ...     gradw = - 2 * X.T.dot(y - y_est) / N + 2 * lambda_ * w
    ...     gradl = ww
    ...     return obj, [gradw, gradl]

    Augment the SGD optimizer to take structured inputs

    >>> new_sgd = structured_sgd(sgd)

    Data

    >>> y = np.linspace(1, 10, 100) + np.random.randn(100) + 1
    >>> X = np.array([np.ones(100), np.linspace(1, 100, 100)]).T
    >>> data = np.hstack((y[:, np.newaxis], X))

    Initial values

    >>> w_0 = Parameter(np.array([1., 1.]), Bound())
    >>> lambda_0 = Parameter(.25, Bound())

    >>> res = new_sgd(cost, [w_0, lambda_0], data, batch_size=10,
    ...               eval_obj=True)
    >>> res_w, res_lambda = res.x
    """
    @wraps(sgd)
    def new_sgd(fun, parameters, data, eval_obj=False, **sgd_kwargs):

        (array1d, fbounds), shapes = flatten(parameters,
                                             hstack=bt.hstack,
                                             shape=bt.shape,
                                             ravel=bt.ravel
                                             )

        flatten_args_dec = flatten_args(shapes)
        new_fun = flatten_args_dec(fun)

        if bool(eval_obj):
            new_fun = flatten_func_grad(new_fun)
        else:
            new_fun = flatten_grad(new_fun)

        result = sgd(new_fun, array1d, data=data, bounds=fbounds,
                     eval_obj=eval_obj, **sgd_kwargs)

        result['x'] = tuple(unflatten(result['x'], shapes))
        return result

    return new_sgd


def logtrick_minimizer(minimizer):
    """
    Log-Trick decorator for optimizers.

    This decorator implements the "log trick" for optimizing positive bounded
    variables. It will apply this trick for any variables that correspond to a
    Positive() bound.

    Examples
    --------
    >>> from scipy.optimize import minimize as sp_min
    >>> from ..btypes import Bound, Positive

    This is a simple cost function where we need to enforce particular
    variabled are positive-only bounded.

    >>> def cost(w, lambda_):
    ...     sq_norm = w.T.dot(w)
    ...     return .5 * lambda_ * sq_norm, lambda_ * w

    Lets enforce that the `w` are positive,

    >>> bounds = [Positive(), Positive(), Positive()]
    >>> new_min = logtrick_minimizer(sp_min)

    Initial values

    >>> w_0 = np.array([.5, .1, .2])
    >>> lambda_0 = .25

    >>> res = new_min(cost, w_0, args=(lambda_0,), bounds=bounds,
    ...               method='L-BFGS-B', jac=True)
    >>> res.x >= 0
    array([ True,  True,  True], dtype=bool)

    Note
    ----
    This decorator only works on unstructured optimizers. However, it can be
    use with structured_minimizer, so long as it is the inner wrapper.
    """
    @wraps(minimizer)
    def new_minimizer(fun, x0, jac=True, bounds=None, **minimizer_kwargs):

        if bounds is None:
            return minimizer(fun, x0, jac=jac, bounds=bounds,
                             **minimizer_kwargs)

        logx, expx, gradx, bounds = logtrick_gen(bounds)

        # Intercept gradient
        if callable(jac):
            def new_jac(x, *fargs, **fkwargs):
                return gradx(jac(expx(x), *fargs, **fkwargs), x)
        else:
            new_jac = jac

        # Intercept objective
        if (not callable(jac)) and bool(jac):
            def new_fun(x, *fargs, **fkwargs):
                o, g = fun(expx(x), *fargs, **fkwargs)
                return o, gradx(g, x)
        else:
            def new_fun(x, *fargs, **fkwargs):
                return fun(expx(x), *fargs, **fkwargs)

        # Transform the final result
        result = minimizer(new_fun, logx(x0), jac=new_jac, bounds=bounds,
                           **minimizer_kwargs)
        result['x'] = expx(result['x'])
        return result

    return new_minimizer


def logtrick_sgd(sgd):
    """
    Log-Trick decorator for stochastic gradients.

    This decorator implements the "log trick" for optimizing positive bounded
    variables using SGD. It will apply this trick for any variables that
    correspond to a Positive() bound.

    Examples
    --------
    >>> from ..optimize import sgd
    >>> from ..btypes import Bound, Positive

    This is a simple cost function where we need to enforce particular
    variabled are positive-only bounded.

    >>> def cost(w, data, lambda_):
    ...     N = len(data)
    ...     y, X = data[:, 0], data[:, 1:]
    ...     y_est = X.dot(w)
    ...     ww = w.T.dot(w)
    ...     obj = (y - y_est).sum() / N + lambda_ * ww
    ...     gradw = - 2 * X.T.dot(y - y_est) / N + 2 * lambda_ * w
    ...     return obj, gradw

    Lets enforce that the `w` are positive,

    >>> bounds = [Positive(), Positive()]
    >>> new_sgd = logtrick_sgd(sgd)

    Data

    >>> y = np.linspace(1, 10, 100) + np.random.randn(100) + 1
    >>> X = np.array([np.ones(100), np.linspace(1, 100, 100)]).T
    >>> data = np.hstack((y[:, np.newaxis], X))

    Initial values

    >>> w_0 = np.array([1., 1.])
    >>> lambda_0 = .25

    >>> res = new_sgd(cost, w_0, data, args=(lambda_0,), bounds=bounds,
    ...               batch_size=10, eval_obj=True)
    >>> res.x >= 0
    array([ True,  True], dtype=bool)

    Note
    ----
    This decorator only works on unstructured optimizers. However, it can be
    use with structured_minimizer, so long as it is the inner wrapper.
    """
    @wraps(sgd)
    def new_sgd(fun, x0, data, bounds=None, eval_obj=False, **sgd_kwargs):

        if bounds is None:
            return sgd(fun, x0, data, bounds=bounds, eval_obj=eval_obj,
                       **sgd_kwargs)

        logx, expx, gradx, bounds = logtrick_gen(bounds)

        if bool(eval_obj):
            def new_fun(x, *fargs, **fkwargs):
                o, g = fun(expx(x), *fargs, **fkwargs)
                return o, gradx(g, x)
        else:
            def new_fun(x, *fargs, **fkwargs):
                return gradx(fun(expx(x), *fargs, **fkwargs), x)

        # Transform the final result
        result = sgd(new_fun, logx(x0), data, bounds=bounds, eval_obj=eval_obj,
                     **sgd_kwargs)
        result['x'] = expx(result['x'])
        return result

    return new_sgd


#
# Helper functions
#

def logtrick_gen(bounds):
    """Generate warping functions and new bounds for the log trick."""
    # Test which parameters we can apply the log trick too
    ispos = np.array([isinstance(b, bt.Positive) for b in bounds], dtype=bool)
    nispos = ~ispos

    # Functions that implement the log trick
    def logx(x):
        xwarp = np.empty_like(x)
        xwarp[ispos] = np.log(x[ispos])
        xwarp[nispos] = x[nispos]
        return xwarp

    def expx(xwarp):
        x = np.empty_like(xwarp)
        x[ispos] = np.exp(xwarp[ispos])
        x[nispos] = xwarp[nispos]
        return x

    def gradx(grad, xwarp):
        gwarp = np.empty_like(grad)
        gwarp[ispos] = grad[ispos] * np.exp(xwarp[ispos])
        gwarp[nispos] = grad[nispos]
        return gwarp

    # Redefine bounds as appropriate for new ranges for numerical stability
    for i, (b, pos) in enumerate(zip(bounds, ispos)):
        if pos:
            upper = EXPMAX if b.upper is None else np.log(b.upper)
            bounds[i] = bt.Bound(lower=LOGMINPOS, upper=upper)

    return logx, expx, gradx, bounds


def flatten_grad(func):
    """
    Decorator to flatten structured gradients.

    Examples
    --------
    >>> def cost(w, lambda_):
    ...     sq_norm = w.T.dot(w)
    ...     return lambda_ * w, .5 * sq_norm
    >>> grad = cost(np.array([.5, .1, -.2]), .25)

    >>> len(grad)
    2
    >>> grad_w, grad_lambda = grad
    >>> np.shape(grad_w)
    (3,)
    >>> np.shape(grad_lambda)
    ()
    >>> grad_w
    array([ 0.125,  0.025, -0.05 ])
    >>> np.isclose(grad_lambda, 0.15)
    True

    >>> cost_new = flatten_grad(cost)
    >>> grad_new = cost_new(np.array([.5, .1, -.2]), .25)
    >>> grad_new
    array([ 0.125,  0.025, -0.05 ,  0.15 ])
    """
    @wraps(func)
    def new_func(*args, **kwargs):
        return flatten(func(*args, **kwargs), returns_shapes=False)

    return new_func


def flatten_func_grad(func):
    """
    Decorator to flatten structured gradients and return objective.

    Examples
    --------
    >>> def cost(w, lambda_):
    ...     sq_norm = w.T.dot(w)
    ...     return .5 * lambda_ * sq_norm, [lambda_ * w, .5 * sq_norm]
    >>> val, grad = cost(np.array([.5, .1, -.2]), .25)

    >>> np.isclose(val, 0.0375)
    True

    >>> len(grad)
    2
    >>> grad_w, grad_lambda = grad
    >>> np.shape(grad_w)
    (3,)
    >>> np.shape(grad_lambda)
    ()
    >>> grad_w
    array([ 0.125,  0.025, -0.05 ])
    >>> np.isclose(grad_lambda, 0.15)
    True

    >>> cost_new = flatten_func_grad(cost)
    >>> val_new, grad_new = cost_new(np.array([.5, .1, -.2]), .25)
    >>> val == val_new
    True
    >>> grad_new
    array([ 0.125,  0.025, -0.05 ,  0.15 ])
    """
    @wraps(func)
    def new_func(*args, **kwargs):
        val, grad = func(*args, **kwargs)
        return val, flatten(grad, returns_shapes=False)

    return new_func


def flatten_args(shapes):
    """
    Decorator to flatten structured arguments to a function.

    Examples
    --------
    >>> @flatten_args([(5,), ()])
    ... def f(w, lambda_):
    ...     return .5 * lambda_ * w.T.dot(w)
    >>> np.isclose(f(np.array([2., .5, .6, -.2, .9, .2])), .546)
    True
    >>> w = np.array([2., .5, .6, -.2, .9])
    >>> lambda_ = .2
    >>> np.isclose(.5 * lambda_ * w.T.dot(w), .546)
    True

    Some other curious applications
    >>> from operator import mul
    >>> flatten_args_dec = flatten_args([(), (3,)])
    >>> func = flatten_args_dec(mul)
    >>> func(np.array([3.1, .6, 1.71, -1.2]))
    array([ 1.86 ,  5.301, -3.72 ])
    >>> 3.1 * np.array([.6, 1.71, -1.2])
    array([ 1.86 ,  5.301, -3.72 ])
    >>> flatten_args_dec = flatten_args([(9,), (15,)])
    >>> func = flatten_args_dec(np.meshgrid)
    >>> x, y = func(np.arange(-5, 7, .5)) # 7 - (-5) / 0.5 = 24 = 9 + 15
    >>> x.shape
    (15, 9)
    >>> x[0, :]
    array([-5. , -4.5, -4. , -3.5, -3. , -2.5, -2. , -1.5, -1. ])
    >>> y.shape
    (15, 9)
    >>> y[:, 0]
    array([-0.5,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,
            5. ,  5.5,  6. ,  6.5])
    """
    def flatten_args_dec(func):

        @wraps(func)
        def new_func(array1d, *args, **kwargs):
            args = tuple(unflatten(array1d, shapes)) + args
            return func(*args, **kwargs)

        return new_func

    return flatten_args_dec
