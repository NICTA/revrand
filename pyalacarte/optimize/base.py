import numpy as np

from scipy.optimize import minimize as sp_min
from .nlopt_wrap import minimize as nl_min

def minimize(fun, x0, args=None, method=None, bounds=None, ftol=None,
             xtol=None, maxiter=None, jac=True):
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

def sp_minimize(fun, x0, args, method, bounds, ftol, maxiter, jac):

    if args is None:
        args = ()

    options = {}
    if maxiter:
        options['maxiter'] = maxiter

    return sp_min(fun, x0, args, method=method, jac=jac, tol=ftol,
                options=options, bounds=bounds)