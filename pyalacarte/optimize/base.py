import numpy as np

from warnings import warn
from scipy.optimize import minimize as sp_min

def minimize(fun, x0, args=(), method=None, jac=None, bounds=[], 
             constraints=[], use_nlopt=False, **options):
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
    if use_nlopt:
    
        try:
            from .nlopt_wrap import minimize as nl_min
        except ImportError:
            warn("NLopt could not be imported. Defaulting to scipy.optimize")
        else:
            return nl_min(fun, x0, args=args, method=method, jac=jac, bounds=bounds, 
                      constraints=constraints, options=options)

    return sp_min(fun, x0, args=args, method=method, jac=jac, bounds=bounds, 
                      constraints=constraints, options=options)
