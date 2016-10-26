"""Optimize Base Decorators."""
import logging
from functools import partial
from itertools import chain
from six import wraps

import numpy as np

import revrand.btypes as bt
from ..utils import flatten, unflatten, map_recursive
from .sgd import gen_subset

# Set up logging
log = logging.getLogger(__name__)


# Constants
MINPOS = 1e-100  # Min for log trick warped data
MAXPOS = np.sqrt(np.finfo(float).max)  # Max for log trick warped data
LOGMINPOS = np.log(MINPOS)
EXPMAX = np.log(MAXPOS)
NSET = 1000  # For random sampling dataset SGD parameter sampling


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
    def new_minimizer(fun, parameters, jac=True, args=(), n_starts=0,
                      random_state=None, **minimizer_kwargs):

        (array1d, fbounds), shapes = flatten(
            parameters,
            hstack=bt.hstack,
            shape=bt.shape,
            ravel=partial(bt.ravel, random_state=random_state)
        )

        # Find best random starting candidate if we are doing random starts
        if n_starts > 0:
            array1d = __random_starts(
                fun=fun,
                parameters=parameters,
                jac=jac,
                args=args,
                n_starts=n_starts,
                random_state=random_state
            )

        # Wrap function calls to work with wrapped minimizer
        flatten_args_dec = flatten_args(shapes)
        new_fun = flatten_args_dec(fun)

        # Wrap gradient calls to work with wrapped minimizer
        if callable(jac):
            new_jac = flatten_args_dec(jac)
        else:
            new_jac = jac
            if bool(jac):
                new_fun = flatten_func_grad(new_fun)

        result = minimizer(new_fun, array1d, jac=new_jac, args=args,
                           bounds=fbounds, **minimizer_kwargs)
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
    def new_sgd(fun, parameters, data, eval_obj=False, args=(), n_starts=100,
                random_state=None, **sgd_kwargs):

        (array1d, fbounds), shapes = flatten(parameters,
                                             hstack=bt.hstack,
                                             shape=bt.shape,
                                             ravel=bt.ravel
                                             )

        flatten_args_dec = flatten_args(shapes)
        new_fun = flatten_args_dec(fun)

        # Find best random starting candidate if we are doing random starts
        if eval_obj and n_starts > 0:
            array1d = __random_starts(
                fun=fun,
                parameters=parameters,
                jac=True,
                args=tuple(chain(gen_subset(data, NSET, random_state), args)),
                n_starts=n_starts,
                random_state=random_state
            )

        if bool(eval_obj):
            new_fun = flatten_func_grad(new_fun)
        else:
            new_fun = flatten_grad(new_fun)

        result = sgd(new_fun, array1d, data=data, bounds=fbounds, args=args,
                     eval_obj=eval_obj, random_state=random_state,
                     **sgd_kwargs)

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


#
# Private module functions
#

def __random_starts(fun, parameters, jac, args, n_starts, random_state):

    if n_starts < 1:
        raise ValueError("n_starts has to be greater than or equal to 1")

    log.info("Evaluating random starts...")

    # Deal with gradient returns from objective function
    if jac is True:
        call_fun = lambda *fargs: fun(*fargs)[0]

    # No gradient returns or jac is callable
    else:
        call_fun = fun

    # Randomly draw parameters and evaluate function
    def fun_eval():
        params = map_recursive(lambda p: p.rvs(random_state), parameters,
                               output_type=list)
        obj = call_fun(*chain(params, args))
        return obj, params

    # Test randomly drawn parameters
    sample_gen = (fun_eval() for _ in range(n_starts))
    obj, params = min(sample_gen, key=lambda t: t[0])

    log.info("Best start found with objective = {}".format(obj))

    return flatten(params, returns_shapes=False)
