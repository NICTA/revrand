"""
Reusable decorators
"""

from ..utils.base import flatten, unflatten
from collections import OrderedDict
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


def flatten_args(fn):
    """
    Examples
    --------
    >>> @flatten_args
    ... def f(x):
    ...     return 2*x

    >>> x, y, z = f(np.array([1., 2.]), 3., np.array([[1., 2.],[.5, .9]]))

    >>> x
    array([ 2.,  4.])

    >>> y
    6.0

    >>> z
    array([[ 2. ,  4. ],
           [ 1. ,  1.8]])
    """
    @wraps(fn)
    def new_fn(*args):
        args_flat, shapes = flatten(args)
        result = fn(args_flat)
        return unflatten(result, shapes=shapes)

    return new_fn


def vectorize_args(fn):
    """
    When defining functions of several variables, it is usually more 
    readable to write out each variable as a separate argument. This is 
    also convenient for evaluating functions on a `numpy.meshgrid`. 

    However, the family of optimizers in `scipy.optimize` expects that 
    all functions, including those of several variables, receive a 
    single argument, which is a `numpy.ndarray` in the case of functions
    of several variables.

    Readability counts. We need not compromise readability to conform to
    some interface when higher-order functions/decorators can abstract 
    away the details for us. This is what this decorator does. 

    See Also
    --------
    revrand.utils.decorators.unvectorize_args

    Examples
    --------

    Optimizers such as those in `scipy.optimize` expects a function 
    defined like this.

    >>> def fun1(v):
    ...     # elliptic parabaloid
    ...     return 2*v[0]**2 + 2*v[1]**2 - 4
    
    >>> a = np.array([2, 3])
    
    >>> fun1(a)
    22

    Whereas this representation is not only more readable but more 
    natural.

    >>> def fun2(x, y):
    ...     # elliptic parabaloid
    ...     return 2*x**2 + 2*y**2 - 4
    
    >>> fun2(2, 3)
    22

    It is also important for evaluating functions on a `numpy.meshgrid`

    >>> y, x = np.mgrid[-5:5:0.2, -5:5:0.2]
    >>> fun2(x, y)
    array([[ 96.  ,  92.08,  88.32, ...,  84.72,  88.32,  92.08],
           [ 92.08,  88.16,  84.4 , ...,  80.8 ,  84.4 ,  88.16],
           [ 88.32,  84.4 ,  80.64, ...,  77.04,  80.64,  84.4 ],
           ..., 
           [ 84.72,  80.8 ,  77.04, ...,  73.44,  77.04,  80.8 ],
           [ 88.32,  84.4 ,  80.64, ...,  77.04,  80.64,  84.4 ],
           [ 92.08,  88.16,  84.4 , ...,  80.8 ,  84.4 ,  88.16]])

    We can easily reconcile the differences between these representation
    without having to compromise readability.

    >>> fun1(a) == vectorize_args(fun2)(a)
    True

    >>> @vectorize_args
    ... def fun3(x, y):
    ...     # elliptic parabaloid
    ...     return 2*x**2 + 2*y**2 - 4
    
    >>> fun1(a) == fun3(a)
    True
    """
    @wraps(fn)
    def new_fn(vec):
        return fn(*vec)
    return new_fn

def unvectorize_args(fn):

    """
    See Also
    --------
    revrand.utils.decorators.vectorize_args

    Examples
    --------
    The Rosenbrock function is commonly used as a performance test 
    problem for optimization algorithms. It and its derivatives are 
    included in `scipy.optimize` and is implemented as expected by the 
    family of optimization methods in `scipy.optimize`.

        def rosen(x):
            return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    This representation makes it unwieldy to perform operations such as 
    plotting since it is less straightforward to evaluate the function 
    on a `meshgrid`. This decorator helps reconcile the differences 
    between these representations.

    >>> from scipy.optimize import rosen

    >>> rosen(np.array([0.5, 1.5]))
    156.5

    >>> unvectorize_args(rosen)(0.5, 1.5) 
    ... # doctest: +NORMALIZE_WHITESPACE
    156.5    

    The `rosen` function is implemented in such a way that it 
    generalizes to the Rosenbrock function of any number of variables. 
    This decorator supports can support any functions defined in a 
    similar manner.

    The function with any number of arguments are well-defined:

    >>> rosen(np.array([0.5, 1.5, 1., 0., 0.2]))
    418.0

    >>> unvectorize_args(rosen)(0.5, 1.5, 1., 0., 0.2)
    ... # can accept any variable number of arguments!
    418.0

    Make it easier to work with for other operations

    >>> rosen_ = unvectorize_args(rosen)
    >>> y, x = np.mgrid[0:2.1:0.05, -1:1.2:0.05]
    >>> z = rosen_(x, y)
    >>> z.round(2)
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
    @wraps(fn)
    def new_fn(*args):
        return fn(np.asarray(args))
    return new_fn

def vectorize_result(fn):
    @wraps(fn)
    def new_fn(*args):
        return np.asarray(fn(*args))
    return new_fn
