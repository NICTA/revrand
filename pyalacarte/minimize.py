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


def sgd(fun, x0, Data, args=(), bounds=None, batchsize=100, rate=1.0,
        gtol=1e-3, maxiter=1e6, eval_obj=False):
    """ Stochastic Gradient Descent, using Adagrad for the learning rate.

        Arguments:
            fun: the function to evaluate, this must have the signature
                `[obj,] grad = fun(x, Data, ...)`, where the `eval_obj`
                argument tells `sgd` if an objective function value is going to
                be returned by `fun`.
            x0: a sequence/1D array of initial values for the parameters to
                learn.
            Data: a numpy array or sequence of data to input into `fun`. This
                will be split along the first axis (axis=0), and then input
                into `fun`.
            args: an optional sequence of arguments to give to fun.
            bounds: sequence, optional. Bounds for variables, (min, max) pairs
                for each element in x, defining the bounds on that parameter.
                Use None for one of min or max when there is no bound in that
                direction.
            batchsize: (int), optional. The number of observations in an SGD
                batch.
            rate: (float), optional. The learning rate of SGD.
            gtol: (float), optional. The norm of the gradient of x that will
                trigger a convergence condition.
            maxiter: (int) optional. Maximum number of mini-batch function
                evaluations before termination.
            eval_obj: (bool), optional. This indicates whether or not `fun`
                also evaluates and returns the objective function value. If
                this is true, `fun` must return `(obj, grad)` and then a list
                of objective function values is also returned.

        Returns:
            (dict): with members:

                'x' (array): the final result
                'norms' (list): the list of gradient norms
                'message' (str): the convergence condition ('converge' or
                    'maxiter')
                'objs' (list): the list of objective function evaluations of
                    `eval_obj` is True.
                'fun' (float): the final objective function evaluation if
                    `eval_obj` is True.
    """

    N = Data.shape[0]
    x = np.array(x0, copy=True, dtype=float)
    D = x.shape[0]

    # Make sure we have a valid batch size
    if N < batchsize:
        batchsize = N

    # Process bounds
    if bounds is not None:
        lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])

        if len(lower) != D:
            raise ValueError("The dimension of the bounds does not match x0!")

    # Initialise
    batchgen = _sgd_batches(N, batchsize)
    gnorm = np.inf
    Gsum = np.zeros_like(x0)
    it = 0

    # Learning Records
    if eval_obj:
        objs = []
    norms = []

    while (it < maxiter) and (gnorm > gtol):

        if not eval_obj:
            grad = fun(x, Data[next(batchgen)], *args)
        else:
            obj, grad = fun(x, Data[next(batchgen)], *args)

        # Truncate gradients if bounded
        if bounds is not None:
            xlower = x <= lower
            grad[xlower] = np.minimum(grad[xlower], 0)
            xupper = x >= upper
            grad[xupper] = np.maximum(grad[xupper], 0)

        Gsum += np.power(grad, 2)
        x -= rate * grad / np.sqrt(Gsum)

        # if it == 10:
        #     Gsum[Gsum > 100 * grad**2] = 0

        # print("sgd:", x[-2], grad[-2], Gsum[-2])
        # import time; time.sleep(0.5)

        # Trucate steps if bounded
        if bounds is not None:
            x = np.minimum(np.maximum(x, lower), upper)

        gnorm = np.linalg.norm(grad)
        norms.append(gnorm)
        if eval_obj:
            objs.append(obj)

        it += 1

    # Format results
    res = {'x': x,
           'norms': norms,
           'message': 'converge' if it < maxiter else 'maxiter'
           }

    if eval_obj:
        res['objs'] = objs
        res['fun'] = obj

    return res


def _trunc_grad(x, grad, gradssum, rate, bound, condition):

    x[condition] = bound[condition]
    gradssum[condition] -= grad[condition]**2

    grad[condition] = 0  #x[condition] - bound[condition]
        # * np.sqrt(gradssum[condition]) / rate
    # gradssum[condition] += grad[condition]**2

    return x, grad, gradssum


def _sgd_batches(N, batchsize):
    """ Batch index generator for SGD that will yeild random batches, and touch
        all of the data (given sufficient interations).

        Arguments:
            N, (int): length of dataset.
            batchsize, (int): number of data points in each batch.

        Yields:
            array: of size (batchsize,) of random (int).
    """

    while True:
        batch_inds = np.array_split(np.random.permutation(N),
                                    round(N / batchsize))

        for b_inds in batch_inds:
            yield b_inds


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
            obj, grad[:] = fun(x, *args) if args else fun(x)
        else:
            obj = fun(x, *args) if args else fun(x)

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
