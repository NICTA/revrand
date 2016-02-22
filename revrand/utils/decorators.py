"""Reusable decorators."""

from ..utils.base import flatten, unflatten
from collections import OrderedDict
from itertools import repeat
from six import wraps

import numpy as np


class Memoize(dict):
    """
    Examples
    --------
    >>> @Memoize
    ... def fib(n):
    ...     if n < 2:
    ...         return n
    ...     return fib(n-2) + fib(n-1)

    >>> fib(10)
    55

    >>> isinstance(fib, dict)
    True

    >>> fib == {
    ...     (0,):  0,
    ...     (1,):  1,
    ...     (2,):  1,
    ...     (3,):  2,
    ...     (4,):  3,
    ...     (5,):  5,
    ...     (6,):  8,
    ...     (7,):  13,
    ...     (8,):  21,
    ...     (9,):  34,
    ...     (10,): 55,
    ... }
    True

    Order is not necessarily maintained.

    >>> sorted(fib.keys())
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]

    >>> sorted(fib.values())
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    def __init__(self, func):
        self.func = func
        super(Memoize, self).__init__()

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        self[key] = self.func(*key)
        return self[key]


class OrderedMemoize(Memoize, OrderedDict):
    """
    Examples
    --------
    >>> @OrderedMemoize
    ... def fib(n):
    ...     if n < 2:
    ...         return n
    ...     return fib(n-2) + fib(n-1)

    >>> fib(10)
    55

    The arguments and values are cached in the order they were called.

    >>> list(fib.keys())
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]

    >>> list(fib.values())
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    >>> fib # doctest: +NORMALIZE_WHITESPACE
    OrderedMemoize([((0,), 0),
                    ((1,), 1),
                    ((2,), 1),
                    ((3,), 2),
                    ((4,), 3),
                    ((5,), 5),
                    ((6,), 8),
                    ((7,), 13),
                    ((8,), 21),
                    ((9,), 34),
                    ((10,), 55)])
    """
    pass


def flatten_args(func, shapes=None, order='C'):
    """
    Examples
    --------
    >>> def f(w, lambda_):
    ...     return .5 * lambda_ * w.T.dot(w)
    >>> g = flatten_args(f, shapes=[(5,), ()])
    >>> np.isclose(g(np.array([2., .5, .6, -.2, .9, .2])),
    ...            f(np.array([2., .5, .6, -.2, .9]), .2))
    True

    >>> from scipy.spatial.distance import mahalanobis
    >>> c = np.random.randn(4, 4)
    >>> M = c.dot(c.T)
    >>> u = np.random.randn(4)
    >>> v = np.random.randn(4)
    >>> a, shapes = flatten((u, v, M), returns_shapes=True)
    >>> mahalanobis_flattened = flatten_args(mahalanobis, shapes=shapes)
    >>> np.isclose(mahalanobis(u, v, M), mahalanobis_flattened(a))
    True

    >>> f = lambda x, y: 2*x**2 + 2*y**2 - 4 # elliptic paraboloid
    >>> f_ = flatten_args(f)
    >>> f(2., 1.5) == f_(np.array([2., 1.5]))
    True

    Some other interesting applications:

    >>> from operator import mul
    >>> func = flatten_args(mul, shapes=[(), (3,)])
    >>> np.allclose(func(np.array([3.1, .6, 1.71, -1.2])),
    ...             3.1 * np.array([.6, 1.71, -1.2]))
    True

    >>> func = flatten_args(np.meshgrid, shapes=[(9,), (15,)])
    >>> x, y = func(np.arange(-5, 7, .5)) # 7 - (-5) / 0.5 = 24 = 15 + 9
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

    @wraps(func)
    def new_func(array1d, *args, **kwargs):

        nonlocal shapes
        if shapes is None:
            # equiv to `shapes = [()] * len(array1d)` but IMHO cleaner
            shapes = list(repeat((), len(array1d)))

        args = tuple(unflatten(array1d, shapes, order)) + args
        return func(*args, **kwargs)

    return new_func


def unflatten_args(func, order='C', returns_shapes=False):
    """
    See Also
    --------
    revrand.utils.decorators.flatten_args

    Examples
    --------
    The Rosenbrock function is commonly used as a test problem for optimization
    algorithms. It and its derivatives are included in `scipy.optimize` and is
    implemented as expected by the family of optimization methods in
    `scipy.optimize`.

        def rosen(x):
            return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    This representation makes it unwieldy to perform operations such as
    plotting since it is less straightforward to evaluate the function on a
    `meshgrid`. This decorator helps reconcile the differences between these
    representations.

    >>> from scipy.optimize import rosen
    >>> rosen(np.array([0.5, 1.5]))
    156.5

    >>> unflatten_args(rosen)(0.5, 1.5)
    ... # doctest: +NORMALIZE_WHITESPACE
    156.5

    The `rosen` function is implemented in such a way that it
    generalizes to the Rosenbrock function of any number of variables.
    This decorator supports can support any functions defined in a
    similar manner.

    The function with any number of arguments are well-defined:

    >>> rosen(np.array([0.5, 1.5, 1., 0., 0.2]))
    418.0

    >>> unflatten_args(rosen)(0.5, 1.5, 1., 0., 0.2)
    ... # can accept any variable number of arguments!
    418.0

    Make it easier to work with for other operations

    >>> rosen_ = np.vectorize(unflatten_args(rosen))
    >>> y, x = np.mgrid[0:2.1:0.05, -1:1.2:0.05]
    >>> z = rosen_(x, y)
    >>> z.round(2) # doctest: +NORMALIZE_WHITESPACE
    array([[ 104.  ,   85.25,   69.22, ...,  121.55,  146.42,  174.92],
           [  94.25,   76.48,   61.37, ...,  110.78,  134.57,  161.95],
           [  85.  ,   68.2 ,   54.02, ...,  100.5 ,  123.22,  149.47],
           ...,
           [  94.25,  113.53,  133.57, ...,   71.83,   54.77,   39.4 ],
           [ 104.  ,  124.25,  145.22, ...,   80.55,   62.42,   45.92],
           [ 114.25,  135.48,  157.37, ...,   89.78,   70.57,   52.95]])

    Now this can be directly plotted with `mpl_toolkits.mplot3d.Axes3D`
    and `ax.plot_surface`.

    """
    @wraps(func)
    def new_func(*args, **kwargs):

        if returns_shapes:
            array1d, shapes = flatten(args, order, returns_shapes)
            return func(array1d, **kwargs), shapes

        return func(flatten(args, order, returns_shapes), **kwargs)

    return new_func


def vectorize_result(fn):
    @wraps(fn)
    def new_fn(*args):
        return np.asarray(fn(*args))
    return new_fn