import numpy as np

from ..utils import flatten, unflatten
from ..externals import check_random_state

from itertools import repeat
from warnings import warn
from six import wraps

# MINIMIZE_BACKENDS = [
#     'scipy',
#     'nlopt',
#     'sgd'
# ]


def get_minimize(backend='scipy'):
    """
    >>> minimize = get_minimize() # doctest: +SKIP
    >>> minimize = get_minimize('nlopt') # doctest: +SKIP
    >>> minimize = get_minimize('foo') # doctest: +SKIP
    """
    from .spopt_wrap import minimize as minimize_

    if backend == 'nlopt':
        try:
            from .nlopt_wrap import minimize as minimize_
        except ImportError:
            warn('NLopt could not be imported.')

    warn('Defaulting to scipy.optimize')

    return minimize_


def minimize(fun, x0, args=(), method=None, jac=True, bounds=None,
             constraints=[], backend='scipy', **options):
    """
    Scipy.optimize.minimize-style wrapper for NLopt and scipy's minimize.

        Arguments:
            fun: callable, Objective function.
            x0: ndarray, Initial guess.
            args, (tuple): optional, Extra arguments passed to the objective
                function and its derivatives (Jacobian).
            method, (int), a value from nlopt.SOME_METHOD (e.g.
                nlopt.NL_BOBYQA). if None, nlopt.NL_BOBYQA is used.
            bounds: sequence, optional. Bounds for variables, (min, max) pairs
                for each element in x, defining the bounds on that parameter.
                Use None for one of min or max when there is no bound in that
                direction.
            ftol, (float): optional. Relative difference of objective function
                between subsequent iterations before termination.
            xtol, (float): optional. Relative difference between values, x,
                between subsequent iterations before termination.
            maxiter, (int): optional. Maximum number of function evaluations
                before termination.
            jac: if using a scipy.optimize.minimize, choose whether or not to
                you will be providing gradients or if they should be calculated
                numerically. Otherwise ignored for NLopt.

        Returns:
            x, (ndarray): The solution of the optimization.
            success, (bool): Whether or not the optimizer exited successfully.
            message, (str): Description of the cause of the termination (see
                NLopts documentation for codes).
            fun, (float): Final value of objective function.

    Examples
    --------
    >>> from scipy.optimize import rosen, rosen_der
    >>> x0 = np.array([ 0.875,  0.75 ])
    >>> minimize(rosen, x0, method='LD_LBFGS', jac=rosen_der, backend='nlopt')
    ... # doctest: +SKIP
    >>> minimize(rosen, x0, method='L-BFGS-B', jac=rosen_der) # doctest: +SKIP
    """
    min_ = get_minimize(backend)

    return min_(fun, x0, args=args, method=method, jac=jac, bounds=bounds,
                constraints=constraints, **options)


def candidate_start_points_random(bounds, n_candidates=1000, random_state=None):
    """
    Randomly generate candidate starting points uniformly within a
    hyperrectangle.

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

        lambda bounds, n_candidates=100: np.random.uniform(*zip(*bounds), size=(n_candidates, len(bounds))).T

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

    >>> candidate_start_points_random([(-1., 2.)], n_candidates=5, random_state=1)
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
    """
    Examples
    --------
    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=[5, 3]).shape
    (2, 15)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                             nums=[5, 10, 9]) # doctest: +ELLIPSIS
    array([[-1.   , -1.   , -1.   , ...,  1.5  ,  1.5  ,  1.5  ],
           [-1.5  , -1.5  , -1.5  , ...,  3.   ,  3.   ,  3.   ],
           [ 0.   ,  0.625,  1.25 , ...,  3.75 ,  4.375,  5.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)],
    ...                             nums=[5, 10, 9]).shape
    (3, 450)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5), (1, 5)],
    ...                             nums=[5, 10, 9, 3]) # doctest: +ELLIPSIS
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

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)], nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)]).shape
    (2, 9)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3)], nums=9).shape
    (2, 81)

    >>> candidate_start_points_lattice([(-1, 1.5), (-1.5, 3), (0, 5)], nums=2).shape
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


def flatten_func_grad(func):
    """
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


def flatten_args(shapes, order='C'):
    """
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
        def new_func(array1d):
            args = unflatten(array1d, shapes, order)
            return func(*args)

        return new_func

    return flatten_args_dec


def augment_minimizer(minimizer):
    """
    Examples
    --------
    >>> from scipy.optimize import minimize as sp_min

    Define a cost function that returns a pair. The first element is the cost
    value and the second element is the gradient represented by a tuple. Even
    if the cost is a function of a single variable, the gradient must be a
    tuple containing one element.

    >>> def cost(w, lambda_):
    ...     sq_norm = w.T.dot(w)
    ...     return .5 * lambda_ * sq_norm, (lambda_ * w, .5 * sq_norm)

    Augment the Scipy optimizer to take structured inputs

    >>> new_min = augment_minimizer(sp_min)

    Initial values

    >>> w_0 = np.array([.5, .1, .2])
    >>> lambda_0 = .25

    >>> res = new_min(cost, (w_0, lambda_0), method='L-BFGS-B', jac=True)
    >>> res_w, res_lambda = res.x
    """

    @wraps(minimizer)
    def new_minimizer(fun, ndarrays, jac=True, **minimizer_kwargs):

        array1d, shapes = flatten(ndarrays)
        flatten_args_dec = flatten_args(shapes)

        new_fun = flatten_args_dec(fun)

        if callable(jac):
            jac = lambda *jac_args, **jac_kwargs: flatten(jac(*jac_args,
                                                              **jac_kwargs),
                                                          returns_shapes=False)
        else:
            if bool(jac):
                new_fun = flatten_func_grad(new_fun)

        result = minimizer(new_fun, array1d, jac=jac, **minimizer_kwargs)
        result['x'] = tuple(unflatten(result['x'], shapes))
        result['jac'] = tuple(unflatten(result['jac'], shapes))
        return result

    return new_minimizer
