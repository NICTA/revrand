"""Stochastic Gradient Descent."""

import numpy as np

from scipy.optimize import OptimizeResult


#
# SGD updater classes for different learning rate methods
#

class SGDUpdater:
    """ Base class for SGD learning rate algorithms. """
    name = 'SGD'

    def __call__(self, x, grad):
        """
        Get a new parameter value

        Parameters
        ----------
        x: ndarray
            input parameters to optimise
        grad: ndarray
            gradient of x

        Returns
        -------
        x_new: ndarray
            the new value for x
        """

        delta = x - grad
        return delta


class AdaDelta(SGDUpdater):
    """ AdaDelta Algorithm """
    name = 'ADADELTA'

    def __init__(self, rho=0.95, epsilon=1e-6):
        """
        Construct an AdaDelta updater object.

        Parameters
        ----------
            rho: float, optional
                smoothing/decay rate parameter, must be [0, 1].
            epsilon: float, optional
                "jitter" term to ensure continued learning (should be small).
        """
        
        # TODO: Make these contracts
        if rho < 0 or rho > 1:
            raise ValueError("Decay rate 'rho' must be between 0 and 1!")

        if epsilon <= 0:
            raise ValueError("Constant 'epsilon' must be > 0!")

        self.rho = rho
        self.epsilon = epsilon
        self.Eg2 = 0
        self.Edx2 = 0

    def __call__(self, x, grad):
        """
        Get a new parameter value from AdaDelta

        Parameters
        ----------
        x: ndarray
            input parameters to optimise
        grad: ndarray
            gradient of x

        Returns
        -------
        x_new: ndarray
            the new value for x
        """

        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * grad**2
        dx = - grad * np.sqrt(self.Edx2 + self.epsilon) \
            / np.sqrt(self.Eg2 + self.epsilon)
        self.Edx2 = self.rho * self.Edx2 + (1 - self.rho) * dx**2
        return x + dx


class AdaGrad(SGDUpdater):
    """ AdaGrad Algorithm """

    name = 'ADAGRAD'

    def __init__(self, eta=1, epsilon=1e-6):
        """
        Construct an AdaGrad updater object.

        Parameters
        ----------
            eta: float, optional
                smoothing/decay rate parameter, must be [0, 1].
            epsilon: float, optional
                small constant term to prevent divide-by-zeros
        """

        if eta <= 0:
            raise ValueError("Learning rate 'eta' must be > 0!")

        if epsilon <= 0:
            raise ValueError("Constant 'epsilon' must be > 0!")

        self.eta = eta
        self.epsilon = epsilon
        self.g2_hist = 0

    def __call__(self, x, grad):
        """
        Get a new parameter value from AdaGrad

        Parameters
        ----------
        x: ndarray
            input parameters to optimise
        grad: ndarray
            gradient of x

        Returns
        -------
        x_new: ndarray
            the new value for x
        """

        self.g2_hist += grad**2
        delta = x - self.eta * grad / (self.epsilon + np.sqrt(self.g2_hist))
        return delta


class Momentum(SGDUpdater):
    """ Momentum Algorithm """
    name = 'Momentum'

    def __init__(self, rho=0.5, eta=0.01):
        """
        Construct a Momentum updater object.

        Parameters
        ----------
            rho: float, optional
                smoothing/decay rate parameter, must be [0, 1].
            eta: float, optional
                weight to give to the momentum term
        """

        if eta <= 0:
            raise ValueError("Learning rate 'eta' must be > 0!")

        if rho < 0 or rho > 1:
            raise ValueError("Decay rate 'rho' must be between 0 and 1!")

        self.eta = eta
        self.rho = rho
        self.dx = 0

    def __call__(self, x, grad):
        """
        Get a new parameter value from the Momentum algorithm

        Parameters
        ----------
        x: ndarray
            input parameters to optimise
        grad: ndarray
            gradient of x

        Returns
        -------
        x_new: ndarray
            the new value for x
        """

        self.dx = self.rho * self.dx - self.eta * grad
        delta = x + self.dx
        return delta


#
# SGD minimizer
#

# TODO: batch_size
def sgd(fun, x0, Data, args=(), bounds=None, batchsize=100, passes=10,
        updater=None, gtol=1e-3, eval_obj=False):
    """ Stochastic Gradient Descent, using ADADELTA for setting the learning
        rate.

        Parameters
        ----------
        fun: callable
            the function to evaluate, this must have the signature
                `[obj,] grad = fun(x, Data, ...)`, where the `eval_obj`
                argument tells `sgd` if an objective function value is going to
                be returned by `fun`.
        x0: ndarray
            a sequence/1D array of initial values for the parameters to learn.
        Data: ndarray
            a numpy array or sequence of data to input into `fun`. This will be
            split along the first axis (axis=0), and then input into `fun`.
        args: sequence, optional
            an optional sequence of arguments to give to fun.
        bounds: sequence, optional
            Bounds for variables, (min, max) pairs for each element in x,
            defining the bounds on that parameter.  Use None for one of min or
            max when there is no bound in that direction.
        batchsize: int, optional
            The number of observations in an SGD batch.
        passes: int, optional
            Number of complete passes through the data before optimization
            terminates (unless it converges first).
        updater: SGDUpdater, optional
            The type of gradient update to use, by default this is AdaDelta
        gtol: float, optional
            The norm of the gradient of x that will trigger a convergence
            condition.
        eval_obj: bool, optional
            This indicates whether or not `fun` also evaluates and returns the
            objective function value. If this is true, `fun` must return
            `(obj, grad)` and then a list of objective function values is also
            returned.

        Returns
        -------
        res: OptimizeResult
            x: narray
                the final result
            norms: list
                the list of gradient norms
            message: str
                the convergence condition ('converge' or 'maxiter')
            objs: list
                the list of objective function evaluations if :code:`eval_obj`
                is True.
            fun: float
                the final objective function evaluation if :code:`eval_obj` is
                True.
    """

    # TODO: dictionary lookup?
    if updater is None:
        updater = AdaDelta()

    N = Data.shape[0]
    x = np.array(x0, copy=True, dtype=float)
    D = x.shape[0]

    # Make sure we have a valid batch size
    batchsize = min(batchsize, N)

    # Process bounds
    if bounds is not None:
        if len(bounds) != D:
            raise ValueError("The dimension of the bounds does not match x0!")
        # TODO: use a zip, pairwise logic
        lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])

    # Learning Records
    obj = None
    objs = []
    norms = []
    allpasses = True

    # TODO: Difficult to test. Should we put inner content into function?
    # TODO: ALL to have a go at improving this! (must still pass
    # test_optimize)...
    for _ in range(passes):
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

            # perform update
            x = updater(x, grad)

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

#
# Module Helpers
#


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
    # TODO: just use iter(...)
    for b_inds in batch_inds:
        yield b_inds
