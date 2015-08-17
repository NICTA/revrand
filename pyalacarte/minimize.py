""" This module wraps NLopt methods with an interface based on
    scipy.optimize.minimize.
"""

import nlopt
import numpy as np
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


def sgd(fun, x0, Data, bounds=None, batchsize=100, rate=1.0, gtol=1e-2,
        maxiter=1e6, eval_obj=False):

    # Make sure we have a valid batch size
    N = Data.shape[0]
    x0 = np.asarray(x0)
    D = x0.shape[0]

    if N < batchsize:
        batchsize = N

    # Process bounds
    if bounds is not None:
        lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])

        print(lower, upper)

        if len(lower) != D:
            raise ValueError("The dimension of the bounds does not match x0!")

    # Initialise
    gnorm = np.inf
    Gsums = np.zeros_like(x0)
    x = x0.copy()
    it = 0

    if eval_obj:
        objs = []
    norms = []

    while (it < maxiter) and (gnorm > gtol):

        b_ind = np.random.choice(N, batchsize, replace=False)
        if not eval_obj:
            grad = fun(x, Data[b_ind])
        else:
            obj, grad = fun(x, Data[b_ind])

        Gsums += grad**2
        gnorm = np.linalg.norm(grad)
        x -= rate * grad / np.sqrt(Gsums)
        if bounds is not None:
            x = np.minimum(np.maximum(x, lower), upper)

        norms.append(gnorm)
        if eval_obj:
            objs.append(obj)

        it += 1

    res = {'x': x,
           'norms': norms,
           'message': 'converge' if it < maxiter else 'maxiter'
           }

    if eval_obj:
        res['objs'] = objs
        res['fun'] = obj

    return res


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
