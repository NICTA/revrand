""" This module wraps NLopt methods with an interface based on
    scipy.optimize.minimize.

    Author:     Daniel Steinberg
    Date:       1 Jun 2015
    Institute:  NICTA

"""

import nlopt
from scipy.optimize import minimize as smin


def minimize(fun, x0, args=None, method=None, bounds=None, ftol=None,
             xtol=None, maxiter=None, jac=True):
    """ Scipy.optimize.minimize-style wrapper for NLopt and scipy's minimize.

        Arguments:
            fun: callable, Objective function.
            x0: ndarray, Initial guess.
            args: tuple, optional, Extra arguments passed to the objective
                functionand its derivatives (Jacobian).
            method: int, a value from nlopt.SOME_METHOD (e.g. nlopt.NL_BOBYQA).
                if None, nlopt.NL_BOBYQA is used.
            bounds: sequence, optional. Bounds for variables, (min, max) pairs
                for each element in x, defining the bounds on that parameter.
                Use None for one of min or max when there is no bound in that
                direction.
            ftol: float, optional. Relative difference of objective function
                between subsequent iterations before termination.
            xtol: float, optional. Relative difference between values, x,
                between subsequent iterations before termination.
            maxiter: int optional. Maximum number of function evaluations
                before termination.
            jac: if using a scipy.optimize.minimize, choose whether or not to
                you will be providing gradients or if they should be calculated
                numerically. Otherwise ignored for NLopt.

        Returns:
            x: (ndarray) The solution of the optimization.
            success: (bool) Whether or not the optimizer exited successfully.
            message: (str) Description of the cause of the termination (see
                NLopts documentation for codes).
            fun: (float): Final value of objective function.

        TODO:
            - Incoporate constraints for COBYLA etc
    """

    if method is None:
        method = nlopt.LN_BOBYQA

    if type(method) is int:
        return _nlopt_wrap(fun, x0, args, method, bounds, ftol, maxiter, xtol)
    elif type(method) is str:
        return _scipy_wrap(fun, x0, args, method, bounds, ftol, maxiter, jac)
    else:
        raise ValueError("Type of input not understood, needs to be int or"
                         " str.")


def _scipy_wrap(fun, x0, args, method, bounds, ftol, maxiter, jac):

        if args is None:
            args = ()

        options = {'maxiter': maxiter} if maxiter else {}
        return smin(fun, x0, args, method=method, jac=jac, tol=ftol,
                    options=options, bounds=bounds)


def _nlopt_wrap(fun, x0, args, method, bounds, ftol, maxiter, xtol):

    # Wrap the objective function into something NLopt expects
    def obj(x, grad=None):

        if grad:
            obj, grad[:] = fun(x, args) if args else fun(x)
        else:
            obj = fun(x, args) if args else fun(x)

        return obj

    # Create NLopt object
    N = len(x0)
    opt = nlopt.opt(method, N)
    opt.set_min_objective(obj)

    # Translate the parameter bounds
    if bounds:
        lower = [b[0] if b[0] else -float('inf') for b in bounds]
        upper = [b[1] if b[1] else float('inf') for b in bounds]
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)

    # Stoping Criteria
    if ftol:
        opt.set_ftol_rel(ftol)
    if xtol:
        opt.set_xtol_rel(xtol)
    if maxiter:
        opt.set_maxeval(maxiter)

    # Run and get results
    res = {
        'x': opt.optimize(x0),
        'fun': opt.last_optimum_value(),
        'message': str(opt.last_optimize_result()),
        'success': opt.last_optimize_result() > 0
        }

    return res
