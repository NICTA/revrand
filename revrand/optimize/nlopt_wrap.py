import nlopt

from scipy.optimize import OptimizeResult
from six.moves import filter
from functools import partial
from warnings import warn
from re import search

NLOPT_ALGORITHMS_KEYS = list(filter(partial(search, r'^[GL][ND]_'),
                                    dir(nlopt)))
NLOPT_ALGORITHMS = {k: getattr(nlopt, k) for k in NLOPT_ALGORITHMS_KEYS}
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


def minimize(fun, x0, args=(), method=None, jac=None, bounds=None,
             constraints=[], **options):
    """
    Parameters
    ----------
    fun : callable
        Objective function

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen, rosen_der
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0, method='ld_lbfgs', jac=rosen_der)
    >>> res.success
    True
    >>> res.message
    'Success'
    >>> np.isclose(res.fun, 0)
    True
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> res = minimize(rosen, x0, method='ld_lbfgs', jac=rosen_der,
    ...                ftol_abs=1e-5)
    >>> res.success
    True
    >>> res.message
    'Optimization stopped because ftol_rel or ftol_abs (above) was reached.'

    >>> res = minimize(rosen, x0, method='ld_lbfgs', jac=rosen_der, foo=3)
    Traceback (most recent call last):
        ...
    ValueError: Parameter foo could not be recognized.

    .. todo:: Some sensible way of testing this.

    >>> x0 = np.array([-1., 1.])
    >>> fun = lambda x: - 2*x[0]*x[1] - 2*x[0] + x[0]**2 + 2*x[1]**2
    >>> dfun = lambda x: np.array([2*x[0] - 2*x[1] - 2, - 2*x[0] + 4*x[1]])
    >>> cons = [{'type': 'eq',
    ...           'fun': lambda x: x[0]**3 - x[1],
    ...           'jac': lambda x: np.array([3.*(x[0]**2.), -1.])},
    ...         {'type': 'ineq',
    ...           'fun': lambda x: x[1] - 1,
    ...           'jac': lambda x: np.array([0., 1.])}]
    >>> res = minimize(fun, x0, jac=dfun, method='LD_SLSQP', constraints=cons)
    >>> res.success
    False
    >>> res.message
    'Halted because roundoff errors limited progress. (In this case, the optimization still typically returns a useful result.)'
    >>> res.x.round(2)
    array([ 0.84,  0.6 ])

    >>> cons = [{'type': 'some bogus type',
    ...           'fun': lambda x: x[0]**3 - x[1],
    ...           'jac': lambda x: np.array([3.*(x[0]**2.), -1.])},
    ...         {'type': 'ineq',
    ...           'fun': lambda x: x[1] - 1,
    ...           'jac': lambda x: np.array([0., 1.])}]
    >>> res = minimize(fun, x0, jac=dfun, method='LD_SLSQP', constraints=cons, ftol_abs=1e-20)
    Traceback (most recent call last):
        ...
    ValueError: Constraint type not recognized
    """
    # Create NLopt object
    dim = len(x0)

    if isinstance(method, str):
        method = get_nlopt_enum_by_name(method)

    opt = nlopt.opt(method, dim)

    xs = []

    # Create NLOpt objective function
    obj_fun = make_nlopt_fun(fun, jac, args, xs)
    opt.set_min_objective(obj_fun)

    # Normalize and set parameter bounds
    if bounds:
        lower, upper = zip(*normalize_bounds(bounds))
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)

    # Equality and Inequality Constraints
    for constr in constraints:

        fun = make_nlopt_fun(fun=constr['fun'], 
                             jac=constr.get('jac', False), 
                             args=constr.get('args', ()))

        if constr['type'] == 'eq':
            opt.add_equality_constraint(fun)
        elif constr['type'] == 'ineq':
            opt.add_inequality_constraint(fun)
        elif constr['type'] in ('eq_m', 'ineq_m'): # TODO: Define '_m' as suffix
                                                   # for now. 
            # TODO: Add support for vector/matrix-valued constraints
            raise NotImplementedError('Vector-valued constraints currently '
                                      'not supported.')
        else:
            raise ValueError('Constraint type not recognized')

    # Set other options, e.g. termination criteria
    # This may or may not be a great idea... Time will tell. 
    for option, val in options.items():
        try:
            set_option = getattr(opt, 'set_{option}'.format(option=option))
        except AttributeError:
            raise ValueError('Parameter {option} could not be ' 
                             'recognized.'.format(option=option))
        else:
            set_option(val)

    # Perform the optimization
    try:
        x = opt.optimize(x0)
    except nlopt.RoundoffLimited:
        x = xs[-1]

    return OptimizeResult(
        x=x,
        fun=opt.last_optimum_value(),
        message=get_nlopt_message(opt.last_optimize_result()),
        success=(opt.last_optimize_result() > 0),
    )


def make_nlopt_fun(fun, jac=True, args=(), xs=None):
    """
    Make NLOpt objective function (as specified by the the `NLOpt Python
    interface`_), from SciPy-style objective functions.

    The NLOpt objective functions are far less pleasant to work with and
    are even *required* to have side effects since gradient arrays are
    required to be passed-by-reference and modifed in-place.

    .. _`NLOpt Python interface`:
       http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference#Objective_function
    """
    def nlopt_fun(x, grad):

        if xs is not None:
            xs.append(x.copy())

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

    return nlopt_fun


def get_nlopt_enum_by_name(method_name=None, default=nlopt.LN_BOBYQA):
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
    if method_name is None:
        method_name = 'LN_BOBYQA'

    try:
        return NLOPT_ALGORITHMS[method_name.upper()]
    except KeyError:
        warn('Method {name} could not be found. Defaulting to '
             '{default}'.format(name=method_name, default=default), 
             RuntimeWarning)
        return default


def normalize_bound(bound):
    """
    Examples
    --------
    >>> normalize_bound((2.6, 7.2))
    (2.6, 7.2)

    >>> normalize_bound((None, 7.2))
    (-inf, 7.2)

    >>> normalize_bound((2.6, None))
    (2.6, inf)

    >>> normalize_bound((None, None))
    (-inf, inf)

    This operation is idempotent:

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
    Examples
    --------
    >>> bounds = [(2.6, 7.2), (None, 2), (3.14, None), (None, None)]
    >>> list(normalize_bounds(bounds))
    [(2.6, 7.2), (-inf, 2), (3.14, inf), (-inf, inf)]
    """
    return map(normalize_bound, bounds)

def get_nlopt_message(ret_code):
    """
    >>> get_nlopt_message(nlopt.SUCCESS)
    'Success'
    
    >>> get_nlopt_message(nlopt.INVALID_ARGS)
    'Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).'
    """ 
    return NLOPT_MESSAGES.get(ret_code)