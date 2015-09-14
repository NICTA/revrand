"""
Reusable decorators
"""

import numpy as np 

def vectorize_args(fn):
    """
    When defining functions of several variables, it is usually more readable to 
    write out each variable as a separate argument. This is alsoconvenient for 
    evaluating functions on a `numpy.meshgrid`. 

    However, the family of optimizers in `scipy.optimize` expects that all 
    functions, including those of several variables, receive a single argument, 
    which is a `numpy.ndarray` in the case of functions of several variables.

    Readability counts. We need not compromise readability to conform to some 
    interface when higher-order functions/decorators can abstract away the 
    details for us. This is what this decorator does. 

    See Also
    --------
    pyalacarte.utils.decorators.unvectorize_args

    Examples
    --------

    Optimizers such as those in `scipy.optimize` expects a function defined like
    this.

    >>> def fun1(v):
    ...     # elliptic parabaloid
    ...     return 2*v[0]**2 + 2*v[1]**2 - 4
    
    >>> a = np.array([2, 3])
    
    >>> fun1(a)
    22

    Whereas this representation is not only more readable but more natural.

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

    def new_fn(vec):
        return fn(*vec)
    return new_fn

def unvectorize_args(fn):

    """
    See Also
    --------
    pyalacarte.utils.decorators.vectorize_args

    Examples
    --------
    The Rosenbrock function is commonly used as a performance test problem for
    optimization algorithms. It and its derivatives are included in 
    `scipy.optimize` and is implemented as expected by the family of 
    optimization methods in `scipy.optimize`.

        def rosen(x):
            return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    This representation makes it unwieldy to perform operations such as 
    plotting since it is less straightforward to evaluate the function on a
    `meshgrid`. This decorator helps reconcile the differences between these
    representations.

    >>> from scipy.optimize import rosen

    >>> rosen(np.array([0.5, 1.5]))
    156.5

    >>> unvectorize_args(rosen)(0.5, 1.5) # doctest: +NORMALIZE_WHITESPACE
    156.5    

    The `rosen` function is implemented in such a way that it generalizes to
    the Rosenbrock function of any number of variables. This decorator 
    supports can support any functions defined in a similar manner.

    The function with any number of arguments are well-defined:

    >>> rosen(np.array([0.5, 1.5, 1., 0., 0.2]))
    418.0

    >>> unvectorize_args(rosen)(0.5, 1.5, 1., 0., 0.2)
    418.0

    Make it easier to work with for other operations

    >>> rosen_ = unvectorize_args(rosen)
    >>> y, x = np.mgrid[-1:3.1:0.1, -2:2.2:0.1]
    >>> z = rosen_(x, y)
    >>> z
    array([[ 2509.  ,  2133.62,  1805.6 , ...,  2126.02,  2501.  ,  2928.02],
           [ 2410.  ,  2042.42,  1721.8 , ...,  2034.82,  2402.  ,  2820.82],
           [ 2313.  ,  1953.22,  1640.  , ...,  1945.62,  2305.  ,  2715.62],
           ..., 
           [  153.  ,    74.02,    27.2 , ...,    66.42,   145.  ,   260.42],
           [  130.  ,    58.82,    19.4 , ...,    51.22,   122.  ,   229.22],
           [  109.  ,    45.62,    13.6 , ...,    38.02,   101.  ,   200.02]])

    Now this can be directly plotted with `mpl_toolkits.mplot3d.Axes3D` and 
    `ax.plot_surface`.

    """

    def new_fn(*args):
        return fn(np.asarray(args))
    return new_fn

def vectorize_result(fn):
    def new_fn(*args):
        return np.asarray(fn(*args))
    return new_fn