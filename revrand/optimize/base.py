import numpy as np

from ..utils import flatten, unflatten, flatten_args, flatten_result
from ..externals import check_random_state

from collections import namedtuple
from itertools import repeat
from warnings import warn
from six import wraps

# MINIMIZE_BACKENDS = [
#     'scipy',
#     'nlopt',
#     'sgd'
# ]


class Bound(namedtuple('Bound', ['lower', 'upper'])):
    """
    Define bounds on a variable for the optimiser. This defaults to all
    real values allowed (i.e. no bounds).

    Parameters
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.

    Attributes
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.

    Examples
    --------
    >>> b = Bound(1e-10, upper=1e-5)
    >>> b
    Bound(lower=1e-10, upper=1e-05)
    >>> b.lower
    1e-10
    >>> b.upper
    1e-05
    >>> isinstance(b, tuple)
    True
    >>> tuple(b)
    (1e-10, 1e-05)
    >>> lower, upper = b
    >>> lower
    1e-10
    >>> upper
    1e-05
    >>> Bound(42, 10)
    Traceback (most recent call last):
        ...
    ValueError: lower bound cannot be greater than upper bound!
    """

    def __new__(cls, lower=None, upper=None, shape=()):
        # Shape is unused, but we have to have the same signature as the init
        # We need new because named tuples are immutable

        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError('lower bound cannot be greater than upper '
                                 'bound!')
        return super(Bound, cls).__new__(cls, lower, upper)

    def __init__(self, lower=None, upper=None, shape=()):
        # This init is just for copying this class.

        self.shape = shape

    def flatten(self):

        if self.shape == ():
            return [self]

        cpy = self.__class__(shape=())

        return [cpy for _ in range(np.prod(self.shape))]


class Positive(Bound):
    """
    Define a positive only bound for the optimiser. This may induce the
    'log trick' in the optimiser (when using an appropriate decorator), which
    will ignore the 'smallest' value (but will stay above 0).

    Parameters
    ---------
    lower : float
        The smallest value allowed for the optimiser to evaluate (if
        not using the log trick).

    Examples
    --------
    >>> b = Positive()
    >>> b # doctest: +SKIP
    Positive(lower=1e-14, upper=None)

    Since ``tuple`` (and by extension its descendents) are immutable,
    the lower bound for all instances of ``Positive`` are guaranteed to
    be positive.

    .. admonition::

       Actually this is not totally true. Something like
       ``b._replace(lower=-42)`` would actually thwart this. Should
       delete this method from ``namedtuple`` when inheriting.

    >>> c = Positive(lower=-10)
    Traceback (most recent call last):
        ...
    ValueError: lower bound must be positive!
    """
    def __new__(cls, lower=1e-14, shape=()):

        if lower <= 0:
            raise ValueError('lower bound must be positive!')

        return super(Positive, cls).__new__(cls, lower, None, shape)

    def __getnewargs__(self):
        """Required for pickling!"""
        return (self.lower,)


def get_minimize(backend='scipy'):
    """
    >>> minimize = get_minimize() # doctest: +SKIP
    >>> minimize = get_minimize('nlopt') # doctest: +SKIP
    >>> minimize = get_minimize('foo') # doctest: +SKIP
    """

    if backend == 'nlopt':
        try:
            from .nlopt_wrap import minimize as minimize_
        except ImportError:
            warn('NLopt could not be imported, defaulting to scipy.optimize.')
    else:
        from .spopt_wrap import minimize as minimize_

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


def candidate_start_points_random(bounds, n_candidates=1000,
                                  random_state=None):
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
    """
    Generate candidate starting points on a uniform grid within a
    hyperrectangle.

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
    Allow an optimizer to accept a list of parameters to optimize, rather than
    just a flattened array.

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

    >>> new_min = structured_minimizer(sp_min)

    Initial values

    >>> w_0 = np.array([.5, .1, .2])
    >>> lambda_0 = .25

    >>> res = new_min(cost, (w_0, lambda_0), method='L-BFGS-B', jac=True)
    >>> res_w, res_lambda = res.x
    """

    @wraps(minimizer)
    def new_minimizer(fun, ndarrays, jac=True, bounds=None,
                      **minimizer_kwargs):

        array1d, shapes = flatten(ndarrays)
        new_fun = flatten_args(fun, shapes=shapes)

        fbounds = _flatten_bounds(bounds)

        if callable(jac):
            jac = flatten_result(jac)
        else:
            if bool(jac):
                new_fun = flatten_result(new_fun, 1)

        result = minimizer(new_fun, array1d, jac=jac, bounds=fbounds,
                           **minimizer_kwargs)
        result['x'] = tuple(unflatten(result['x'], shapes))

        if bool(jac):
            result['jac'] = tuple(unflatten(result['jac'], shapes))

        return result

    return new_minimizer


def structured_sgd(sgd):
    """
    Allow stochastic gradients to accept a list of parameters to optimize,
    rather than just a flattened array.

    Examples
    --------
    >>> from ..optimize import sgd

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

    >>> w_0 = np.array([1., 1.])
    >>> lambda_0 = .25

    >>> res = new_sgd(cost, [w_0, lambda_0], data, batchsize=10, eval_obj=True)
    >>> res_w, res_lambda = res.x
    """

    @wraps(sgd)
    def new_sgd(fun, ndarrays, Data, bounds=None, eval_obj=False,
                **sgd_kwargs):

        array1d, shapes = flatten(ndarrays)
        new_fun = flatten_args(fun, shapes=shapes)

        fbounds = _flatten_bounds(bounds)

        if bool(eval_obj):
            new_fun = flatten_result(new_fun, 1)
        else:
            new_fun = flatten(new_fun, returns_shapes=False)

        result = sgd(new_fun, array1d, Data=Data, bounds=fbounds,
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
    >>> from ..optimize import Bound, Positive

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

        logx, expx, gradx, bounds = _logtrick_gen(bounds)

        if callable(jac):
            jac = lambda x, *fargs, **fkwargs: gradx(jac(expx(x), *fargs,
                                                         **fkwargs), x)
        else:
            if bool(jac):
                def new_fun(x, *fargs, **fkwargs):
                    o, g = fun(expx(x), *fargs, **fkwargs)
                    return o, gradx(g, x)
            else:
                def new_fun(x, *fargs, **fkwargs):
                    return fun(expx(x), *fargs, **fkwargs)

        # Transform the final result
        result = minimizer(new_fun, logx(x0), jac=jac, bounds=bounds,
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
    >>> from ..optimize import sgd, Bound, Positive

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
    ...               batchsize=10, eval_obj=True)
    >>> res.x >= 0
    array([ True,  True], dtype=bool)

    Note
    ----
    This decorator only works on unstructured optimizers. However, it can be
    use with structured_minimizer, so long as it is the inner wrapper.
    """

    @wraps(sgd)
    def new_sgd(fun, x0, Data, bounds=None, eval_obj=False, **sgd_kwargs):

        if bounds is None:
            return sgd(fun, x0, Data, bounds=bounds, eval_obj=eval_obj,
                       **sgd_kwargs)

        logx, expx, gradx, bounds = _logtrick_gen(bounds)

        if bool(eval_obj):
            def new_fun(x, *fargs, **fkwargs):
                o, g = fun(expx(x), *fargs, **fkwargs)
                return o, gradx(g, x)
        else:
            def new_fun(x, *fargs, **fkwargs):
                return gradx(fun(expx(x), *fargs, **fkwargs), x)

        # Transform the final result
        result = sgd(new_fun, logx(x0), Data, bounds=bounds, eval_obj=eval_obj,
                     **sgd_kwargs)
        result['x'] = expx(result['x'])
        return result

    return new_sgd


#
# Helper functions
#

def _logtrick_gen(bounds):

    # Test which parameters we can apply the log trick too
    ispos = [(type(b) is Positive) for b in bounds]

    # Functions that implement the log trick
    logx = lambda x: np.array([np.log(xi) if pos else xi
                               for xi, pos in zip(x, ispos)])
    expx = lambda x: np.array([np.exp(xi) if pos else xi
                               for xi, pos in zip(x, ispos)])
    gradx = lambda g, logx: np.array([gi * np.exp(lxi) if pos else gi
                                      for lxi, gi, pos in zip(logx, g, ispos)])

    # Redefine bounds as appropriate for new ranges
    bounds = [Bound() if pos else b for b, pos in zip(bounds, ispos)]

    return logx, expx, gradx, bounds


def _flatten_bounds(bounds):

    if bounds is None:
        return None

    def unwrap(flatb, bounds):
        for b in bounds:
            if type(b) is tuple:
                flatb.append(b)
            elif type(b) is list:
                unwrap(flatb, b)
            else:
                flatb.extend(b.flatten())

    flat_bounds = []
    unwrap(flat_bounds, bounds)

    return flat_bounds
