"""
Distributed Stochastic Gradient Descent using Apache Spark 
"""

import numpy as np
from scipy.optimize import OptimizeResult
import logging
from revrand.optimize.sgd_updater import AdaDelta
from revrand.optimize.sgd import _sgd_pass

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def sgd_spark(fun, x0, Data, args=(), bounds=None, batchsize=100,
        gtol=1e-3, passes=10, eval_obj=False, rate=0.95, eta=1e-6):

    """ Distributed Stochastic Gradient Descent using Spark.
        ADADELTA is used for setting the learning rate.

        Arguments:
            fun: the function to evaluate, this must have the signature
                `[obj,] grad = fun(x, Data, ...)`, where the `eval_obj`
                argument tells `sgd` if an objective function value is going to
                be returned by `fun`.
            x0: a sequence/1D array of initial values for the parameters to
                learn.
            Data: a Spark RDD representing data to input into `fun`. This
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

    updater = AdaDelta(rate, eta)
    return sgd_u_spark(fun, x0, Data, args, updater, bounds,
                       batchsize, gtol, passes, eval_obj)


def sgd_u_spark(fun, x0, Data, args=(), updater=AdaDelta(), bounds=None,
              batchsize=100, gtol=1e-3, passes=10, eval_obj=False):
    """
        Stochastic Gradient Descent, using provided 'updater' for setting
        the learning rate. Defaults to ADADELTA.
    """

    N = Data.count()
    q = Data.getNumPartitions()
    M = N//q
    batchsize /= q
    x = np.array(x0, copy=True, dtype=float)
    D = x.shape[0]

    # Make sure we have a valid batch size
    if M < batchsize:
        batchsize = M

    # Process bounds
    if bounds is not None:
        lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])

        if len(lower) != D:
            raise ValueError("The dimension of the bounds does not match x0!")

    log.info("##### N={}, q={}, M={}, batchsize={}, passes={}"
             .format(N,q,M,batchsize,passes))

    # Learning Records
    obj = None
    objs = []
    norms = []
    allpasses = True

    # Workers should cache the RDD
    Data.cache()

    # Broadcast additional arguments to 'fun'
    bcArgs = Data.context.broadcast(args)

    for p in range(passes):
        for ind in _sgd_pass(M, batchsize):

            # Broadcast the sample indices and the current parameter values
            bcInd = Data.context.broadcast(ind)
            bcX   = Data.context.broadcast(x)

            # Map
            # Sample the RDD partition a evaluate the objective function gradient
            def fgrad(it):
                sample = np.vstack([s for i,s in enumerate(it) if i in bcInd.value])
                yield fun( bcX.value, sample, *bcArgs.value)

            # Reduce
            # Join together results from multiple partitions
            join1 = lambda a, b: a + b
            join2 = lambda a, b: (a[0] + b[0], a[1] + b[1])

            if not eval_obj:
                grad = Data.mapPartitions(fgrad).reduce(join1)
            else:
                obj, grad = Data.mapPartitions(fgrad).reduce(join2)

            grad /= q

            # Truncate gradients if bounded
            if bounds is not None:
                xlower = x <= lower
                grad[xlower] = np.minimum(grad[xlower], 0)
                xupper = x >= upper
                grad[xupper] = np.maximum(grad[xupper], 0)

            x = updater(x, grad)

            # Truncate steps if bounded
            if bounds is not None:
                x = np.minimum(np.maximum(x, lower), upper)

            gnorm = np.linalg.norm(grad)
            norms.append(gnorm)
            if eval_obj:
                objs.append(obj)

            if gnorm <= gtol:
                allpasses = False
                break

    obj_str = ""
    if eval_obj:
        obj_str = ", Obj = {}".format(objs[-5:])
    log.info("##### End: gnorm={}, allpasses={}, p={}{}"
             .format(gnorm, allpasses, p, obj_str))

    # Format results
    res = OptimizeResult(
        x=x,
        norms=norms,
        message='converge' if not allpasses else 'all passes',
        fun=obj,
        objs=objs
    )

    return res

