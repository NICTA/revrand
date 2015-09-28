"""
Stochastic Gradient Descent
"""

import numpy as np
from scipy.optimize import OptimizeResult

def sgd(fun, x0, Data, args=(), bounds=None, batchsize=100, rate=0.9,
        eta=1e-5, gtol=1e-3, passes=10, eval_obj=False):
    """ Stochastic Gradient Descent, using ADADELTA for setting the learning
        rate.

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
            rate, (float): ADADELTA smoothing/decay rate parameter, must be [0,
                1].
            eta, (float): ADADELTA "jitter" term to ensure continued learning
                (should be small).
            gtol, (float): optional. The norm of the gradient of x that will
                trigger a convergence condition.
            passes, (int): Number of complete passes through the data before
                optimization terminates (unless it converges first).
            eval_obj, (bool): optional. This indicates whether or not `fun`
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

    if rate < 0 or rate > 1:
        raise ValueError("rate must be between 0 and 1!")

    if eta <= 0:
        raise ValueError("eta must be > 0!")

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
    gnorm = np.inf
    Eg2 = 0
    Edx2 = 0

    # Learning Records
    obj = None
    objs = []
    norms = []
    allpasses = True

    for p in range(passes):
        for ind in _sgd_pass(N, batchsize):

            if not eval_obj:
                grad = fun(x, Data[ind], *args)
            else:
                obj, grad = fun(x, Data[ind], *args)

            # Truncate gradients if bounded
            if bounds is not None:
                xlower = x <= lower
                grad[xlower] = np.minimum(grad[xlower], 0)
                xupper = x >= upper
                grad[xupper] = np.maximum(grad[xupper], 0)

            # ADADELTA
            Eg2 = rate * Eg2 + (1 - rate) * grad**2
            dx = - grad * np.sqrt(Edx2 + eta) / np.sqrt(Eg2 + eta)
            Edx2 = rate * Edx2 + (1 - rate) * dx**2
            x += dx

            # Trucate steps if bounded
            if bounds is not None:
                x = np.minimum(np.maximum(x, lower), upper)

            gnorm = np.linalg.norm(grad)
            norms.append(gnorm)
            if eval_obj:
                objs.append(obj)

            if gnorm <= gtol:
                allpasses = False
                break

    # Format results
    res = OptimizeResult(
        x=x,
        norms=norms,
        message='converge' if not allpasses else 'all passes',
        fun=obj,
        objs=objs
    )

    return res


def _sgd_batches(N, batchsize):
    """ Batch index generator for SGD that will yeild random batches, and touch
        all of the data (given sufficient interations). This is an infinite
        sequence.

        Arguments:
            N, (int): length of dataset.
            batchsize, (int): number of data points in each batch.

        Yields:
            array: of size (batchsize,) of random (int).
    """

    while True:
        return _sgd_pass(N, batchsize)


def _sgd_pass(N, batchsize):
    """ Batch index generator for SGD that will yeild random batches for a
        single pass through the whole dataset (i.e. a finitie sequence).

        Arguments:
            N, (int): length of dataset.
            batchsize, (int): number of data points in each batch.

        Yields:
            array: of size (batchsize,) of random (int).
    """

    batch_inds = np.array_split(np.random.permutation(N), round(N / batchsize))

    for b_inds in batch_inds:
        yield b_inds
