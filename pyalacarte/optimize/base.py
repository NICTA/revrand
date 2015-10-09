import numpy as np

from ..utils import flatten, unflatten

from scipy.optimize import minimize as sp_min
from six.moves import zip_longest
from collections import Iterable
from functools import partial
from itertools import repeat
from warnings import warn
from six import wraps

def minimize(fun, x0, args=(), method=None, jac=True, bounds=None, 
             constraints=[], use_nlopt=False, **options):
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
    """
    if use_nlopt:
    
        if bounds is None:
            bounds = []
    
        try:
            from .nlopt_wrap import minimize as nl_min
        except ImportError:
            warn("NLopt could not be imported. Defaulting to scipy.optimize")
        else:
            return nl_min(fun, x0, args=args, method=method, jac=jac, 
                          bounds=bounds, constraints=constraints, **options)

    return sp_min(fun, x0, args=args, method=method, jac=jac, bounds=bounds, 
                  constraints=constraints, options=options)

def candidate_start_points_random(bounds, n_candidates=1000):
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
    Equivalent to::

        lambda bounds, n_candidates=100: np.random.uniform(*zip(*bounds), size=(n_candidates, len(bounds))).T
    
    Examples
    --------
    >>> candidate_start_points_random([(-10., -3.5), (-1., 2.)], 
    ...     n_candidates=5)
    ... # doctest: +SKIP
    array([[-7.7165118 , -8.35534484, -3.625521  , -4.02359491, -4.82233003],
           [ 0.9498959 , -0.37492893,  0.16908869, -0.78321786,  1.40738975]])

    >>> candidate_start_points = candidate_start_points_random(
    ...     [(-10., -3.5), (-1., 2.)])

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

    >>> candidate_start_points_random([(-1., 2.)], n_candidates=5) 
    ... # doctest: +SKIP
    array([[ 0.33371234,  1.52775115,  1.51805039,  0.32079371,  0.75478597]])

    Uniformly sample from hyperrectangle:

    >>> candidate_start_points_random([(-10., -3.5), (-1., 2.), (5., 7.), 
    ... (2.71, 3.14)], n_candidates=5) # doctest: +SKIP
    array([[-8.65860645, -6.83830936, -6.66424853, -7.92209109, -8.87889632],
           [ 0.54385109,  0.63564042,  1.43670096, -0.56410552, -0.61085342],
           [ 5.34469192,  6.8235269 ,  6.74123457,  5.26933478,  6.07431495],
           [ 2.89553972,  3.11428126,  2.95325045,  2.95371842,  2.81686667]])
    """
    low, high = zip(*bounds)
    n_dims = len(bounds)
    return np.random.uniform(low, high, (n_candidates, n_dims)).transpose()

def candidate_start_points_grid(bounds, nums=3):
    """
    Examples
    --------
    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3)], nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3)], nums=[5, 3]).shape
    (2, 15)

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5)], 
    ...                             nums=[5, 10, 9]) # doctest: +ELLIPSIS
    array([[-1.   , -1.   , -1.   , ...,  1.5  ,  1.5  ,  1.5  ],
           [-1.5  , -1.5  , -1.5  , ...,  3.   ,  3.   ,  3.   ],
           [ 0.   ,  0.625,  1.25 , ...,  3.75 ,  4.375,  5.   ]])
    
    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5)], 
    ...                             nums=[5, 10, 9]).shape
    (3, 450)

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5), (1, 5)], 
    ...                             nums=[5, 10, 9, 3]) # doctest: +ELLIPSIS
    array([[-1. , -1. , -1. , ...,  1.5,  1.5,  1.5],
           [-1.5, -1.5, -1.5, ...,  3. ,  3. ,  3. ],
           [ 0. ,  0. ,  0. , ...,  5. ,  5. ,  5. ],
           [ 1. ,  3. ,  5. , ...,  1. ,  3. ,  5. ]])
    
    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5), (1, 5)], 
    ...                             nums=[5, 10, 9, 3]).shape
    (4, 1350)

    Third ``num`` is ignored

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3)], nums=[5, 3, 9])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])

    Third bound is ignored

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5)], nums=[5, 3])
    array([[-1.   , -0.375,  0.25 ,  0.875,  1.5  , -1.   , -0.375,  0.25 ,
             0.875,  1.5  , -1.   , -0.375,  0.25 ,  0.875,  1.5  ],
           [-1.5  , -1.5  , -1.5  , -1.5  , -1.5  ,  0.75 ,  0.75 ,  0.75 ,
             0.75 ,  0.75 ,  3.   ,  3.   ,  3.   ,  3.   ,  3.   ]])
    
    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3)]).shape
    (2, 9)

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3)], nums=9).shape
    (2, 81)

    >>> candidate_start_points_grid([(-1, 1.5), (-1.5, 3), (0, 5)], nums=2).shape
    (3, 8)
    """

    if isinstance(nums, int):
        nums = repeat(nums)

    linspaces = [np.linspace(start, end, num) for (start, end), num \
        in zip(bounds, nums)]
    return np.vstack(a.flatten() for a in np.meshgrid(*linspaces))

def minimize_bounded_start(candidates_func=candidate_start_points_random, 
    *candidates_func_args, **candidates_func_kwargs):
    """
    Examples
    --------
    >>> from scipy.optimize import minimize as sp_min, rosen, rosen_der

    >>> @minimize_bounded_start(n_candidates=250)
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)

    >>> rect = [(-1, 1.5), (-.5, 1.5)]

    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> np.allclose(res.x, np.array([ 1.,  1.]))
    True

    >>> np.isclose(res.fun, 0)
    True

    >>> res.start # doctest: +SKIP
    array([ 0.97,  0.96])

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
    
    >>> @minimize_bounded_start(candidate_start_points_grid, nums=[5, 9])
    ... def my_min(fun, x0, *args, **kwargs):
    ...     return sp_min(fun, x0, *args, **kwargs)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> res.start
    array([ 0.875,  0.75 ])

    >>> minimize_bounded_start_dec = minimize_bounded_start(
    ...     candidate_start_points_grid, nums=[5, 9])

    >>> my_min = minimize_bounded_start_dec(sp_min)
    >>> res = my_min(rosen, rect, method='L-BFGS-B', jac=rosen_der)
    >>> res.start
    array([ 0.875,  0.75 ])

    Just to confirm this is the correct starting point:

    >>> candidates = candidate_start_points_grid(rect, nums=[5, 9])
    >>> candidates[:, rosen(candidates).argmin()]
    array([ 0.875,  0.75 ])
    """

    def minimize_bounded_start_dec(minimize_func):

        @wraps(minimize_func)
        def _minimize_bounded_start(fun, x0_bounds, *args, **kwargs):
            candidate_start_points = candidates_func(x0_bounds,
                *candidates_func_args, **candidates_func_kwargs)
            candidate_start_values = fun(candidate_start_points)
            min_start_point_ind = np.argmin(candidate_start_values)
            min_start_point = candidate_start_points[:, min_start_point_ind]
            res = minimize_func(fun, min_start_point, *args, **kwargs)
            res.start = min_start_point
            return res

        return _minimize_bounded_start
    
    return minimize_bounded_start_dec


def augment_minimizer(minimizer):

    def new_minimizer(fun, *ndarrays, **kwargs):
        array1d, shapes = flatten_join(*ndarrays)
        result = minimizer(fun, array1d, **kwargs)
        result['x'] = split_unflatten(result['x'], shapes)
        return result

    return new_minimizer