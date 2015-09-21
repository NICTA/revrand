import numpy as np
import nlopt

from pyalacarte.utils import couple

from scipy.optimize import OptimizeResult
from six.moves import filter, range
from functools import partial
from warnings import warn
from re import search

NLOPT_ALGORITHMS_KEYS = list(filter(partial(search, r'^[GL][ND]_'), dir(nlopt)))

NLOPT_ALGORITHMS = {k:getattr(nlopt, k) for k in NLOPT_ALGORITHMS_KEYS}

NLOPT_MESSAGES = {
    nlopt.SUCCESS: 'Success',
    nlopt.STOPVAL_REACHED: 'Optimization stopped because stopval (above) '
                           'was reached.',
    nlopt.FTOL_REACHED: 'Optimization stopped because ftol_rel or ftol_abs '
                        '(above) was reached.',
    nlopt.XTOL_REACHED: 'Optimization stopped because xtol_rel or xtol_abs ' 
                        '(above) was reached.',
    nlopt.MAXEVAL_REACHED: 'Optimization stopped because maxeval (above) '
                           'was reached.',
    nlopt.MAXTIME_REACHED: 'Optimization stopped because maxtime (above) ' 
                           'was reached.',
    nlopt.FAILURE: 'Failure',
    nlopt.INVALID_ARGS: 'Invalid arguments (e.g. lower bounds are bigger '
                        'than upper bounds, an unknown algorithm was '
                        'specified, etcetera).',
    nlopt.OUT_OF_MEMORY: 'Ran out of memory.',
    nlopt.ROUNDOFF_LIMITED: 'Halted because roundoff errors limited progress. '
                            '(In this case, the optimization still typically '
                            'returns a useful result.)',
    nlopt.FORCED_STOP: "Halted because of a forced termination: the user " 
                       "called nlopt_force_stop(opt) on the optimization's "
                       "nlopt_opt object opt from the user’s objective "
                       "function or constraints."
}

def get_nlopt_enum_by_name(method_name=None):
    """
    Get NLOpt algorithm object by name. If the algorithm is not found, 
    defaults to `nlopt.LN_BOBYQA`.

    Notes
    -----

    From http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#Nomenclature:

        Each algorithm in NLopt is identified by a named constant, which 
        is passed to the NLopt routines in the various languages in 
        order to select a particular algorithm. These constants are 
        mostly of the form `NLOPT_{G,L}{N,D}_xxxx`, where G/L denotes 
        global/local optimization and N/D denotes derivative-free/
        gradient-based algorithms, respectively.

        For example, the NLOPT_LN_COBYLA constant refers to the COBYLA 
        algorithm (described below), which is a local (L) 
        derivative-free (N) optimization algorithm.

        Two exceptions are the MLSL and augmented Lagrangian algorithms, 
        denoted by NLOPT_G_MLSL and NLOPT_AUGLAG, since whether or not 
        they use derivatives (and whether or not they are global, in 
        AUGLAG's case) is determined by what subsidiary optimization 
        algorithm is specified. 

    Equivalent to::

        partial(NLOPT_ALGORITHMS.get, default=nlopt.LN_BOBYQA)

    Examples
    --------
    >>> get_nlopt_enum_by_name('LN_NELDERMEAD') == nlopt.LN_NELDERMEAD
    True

    >>> get_nlopt_enum_by_name('ln_neldermead') == nlopt.LN_NELDERMEAD
    True

    One is permitted to be cavalier with these method names.

    >>> get_nlopt_enum_by_name('ln_NelderMead') == nlopt.LN_NELDERMEAD
    True

    >>> get_nlopt_enum_by_name() == nlopt.LN_BOBYQA
    True

    >>> get_nlopt_enum_by_name('foobar') == nlopt.LN_BOBYQA
    True

    .. todo:: Exceptional cases (low-priority)

    >>> get_nlopt_enum_by_name('G_MLSL') == nlopt.G_MLSL # doctest: +SKIP
    True

    >>> get_nlopt_enum_by_name('AUGLAG') == nlopt.AUGLAG # doctest: +SKIP
    True
    """

    return NLOPT_ALGORITHMS.get(method_name.upper() if method_name is not None \
        else None, nlopt.LN_BOBYQA)

def get_nlopt_by_name(method_name=None, dimensions=None):
    pass

def make_nlopt_obj(fun, args=(), jac=True):

    """
    Make NLOpt objective function (as specified by the the `NLOpt Python 
    interface`_), from SciPy-style objective functions.

    The NLOpt objective functions are far less pleasant to work with and
    are even *required* to have side effects since gradient arrays are
    required to be passed-by-reference and modifed in-place.

    .. _`NLOpt Python interface`: 
       http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference#Objective_function

    Examples
    --------

    .. todo:: 

       Only a few examples are needed here, the edge-cases are only
       useful for unit testing or inclusion in sphinx documentation.

    >>> from scipy.optimize import rosen, rosen_der
    >>> rosen_couple = couple(rosen, rosen_der)
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

    Gradient-based methods

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=rosen_der)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen_couple, jac=True)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    If a callable jacobian `jac` is specified, it will take precedence 
    over the gradient given by a function that returns a tuple with the 
    gradient as its second value.   

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(couple(rosen, lambda x: 2*x), jac=rosen_der)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    If you use a gradient-based optimization method with `jac=True` but
    fail to supply any gradient information, you will receive a 
    `RuntimeWarning` and terrible results.

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=True)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.3,  0.7,  0.8,  1.9,  1.2])

    Likewise, if you *do* supply gradient information, but set `jac=False`
    you will be reminded of the fact that the gradient information is 
    being ignored through a `RuntimeWarning`. 

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen_couple, jac=False)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.3,  0.7,  0.8,  1.9,  1.2])

    Of course, you can use gradient-based optimization and not supply 
    any gradient information at your own discretion. 
    No warning are raised. 

    >>> opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=False)
    >>> opt.set_min_objective(obj_fun)    
    >>> opt.optimize(x0)
    array([ 1.3,  0.7,  0.8,  1.9,  1.2])

    Derivative-free methods

    >>> opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=False)
    >>> opt.set_min_objective(obj_fun)
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    >>> opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=False)
    >>> opt.set_min_objective(obj_fun)
    >>> opt.set_ftol_abs(1e-11)
    >>> np.allclose(np.array([ 1.,  1.,  1.,  1.,  1.]), opt.optimize(x0))
    True
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    When using derivative-free optimization methods, gradient information
    supplied in any form is disregarded without warning.

    >>> opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen, jac=rosen_der)
    >>> opt.set_min_objective(obj_fun)
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    >>> opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    >>> obj_fun = make_nlopt_obj(rosen_couple, jac=True)
    >>> opt.set_min_objective(obj_fun)
    >>> opt.optimize(x0)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.isclose(opt.last_optimum_value(), 0)
    True

    """

    def nlopt_obj_fun(x, grad):

        ret = fun(x, *args)
        grad_temp = None

        if isinstance(ret, tuple):
            val, grad_temp = ret
        else:
            val = ret

        if grad.size > 0:
            if callable(jac):
                grad[:] = jac(x, *args)
            else:            
                if bool(jac):
                    if grad_temp is None:
                        warn('Using gradient-based optimization with '
                            'jac=True, but no gradient information is '
                            'available.', RuntimeWarning)
                    else:
                        grad[:] = grad_temp
                else:
                    if grad_temp is not None:
                        warn('Using gradient-based optimization with '
                            'jac=False, the provided gradient information '
                            'is ignored.', RuntimeWarning)

        return val

    return nlopt_obj_fun

def normalize_bound(bound):
    """
    >>> normalize_bound((2.6, 7.2))
    (2.6, 7.2)

    >>> normalize_bound((None, 7.2))
    (-inf, 7.2)

    >>> normalize_bound((2.6, None))
    (2.6, inf)

    >>> normalize_bound((None, None))
    (-inf, inf)

    This operation is idempotent
    >>> normalize_bound((-float("inf"), float("inf")))
    (-inf, inf)
    """
    min_, max_ = bound
    
    if min_ is None:
        min_ = -float('inf')
    
    if max_ is None:
        max_ = float('inf')

    return min_, max_

def normalize_bounds(bounds=[]):
    """
    >>> bounds = [(2.6, 7.2), (None, 2), (3.14, None), (None, None)]
    >>> list(normalize_bounds(bounds))
    [(2.6, 7.2), (-inf, 2), (3.14, inf), (-inf, inf)]
    """
    return map(normalize_bound, bounds)

def make_nlopt_message(ret_code):
    """
    >>> make_nlopt_message(nlopt.SUCCESS)
    'Success'
    
    >>> make_nlopt_message(nlopt.INVALID_ARGS)
    'Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).'
    """ 
    return NLOPT_MESSAGES.get(ret_code)

def nlopt_minimize(fun, x0, args=(), method=None, jac=None, bounds=[], 
                   constraints=[], ftol=None, xtol=None, maxiter=None):
    """
    >>> from scipy.optimize import rosen, rosen_der
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    
    >>> res = nlopt_minimize(rosen, x0, method='ld_lbfgs', jac=rosen_der)
    >>> res.success
    True
    >>> res.message
    'Success'
    >>> np.isclose(res.fun, 0)
    True
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> res = nlopt_minimize(rosen, x0, method='cobyla', constraints=[3])
    """
    if constraints:
        warn("Equality and inequality constraints are currently "
             "not supported", NotImplementedError)

    # Create NLopt object
    dim = len(x0)

    if isinstance(method, str):
        method = get_nlopt_enum_by_name(method)

    opt = nlopt.opt(method, dim)

    # Create NLOpt objective function
    obj_fun = make_nlopt_obj(fun, args, jac)
    opt.set_min_objective(obj_fun)

    # Normalize and set parameter bounds
    if bounds:
        lower, upper = zip(*normalize_bounds(bounds))
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)

    # Termination Criteria
    if ftol is not None:
        opt.set_ftol_rel(ftol)

    if xtol is not None:
        opt.set_xtol_rel(xtol)

    if maxiter is not None:
        opt.set_maxeval(maxiter)

    return OptimizeResult(
        x = opt.optimize(x0), 
        fun = opt.last_optimum_value(),
        message = make_nlopt_message(opt.last_optimize_result()),
        success = opt.last_optimize_result() > 0,
    )