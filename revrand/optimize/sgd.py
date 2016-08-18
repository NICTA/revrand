"""Stochastic Gradient Descent."""

from itertools import chain

import numpy as np
from scipy.optimize import OptimizeResult

from ..utils import issequence, endless_permutations


#
# SGD updater classes for different learning rate methods
#

class SGDUpdater:
    """
    Base class for SGD learning rate algorithms.

    Parameters
    ----------
    eta: float, optional
        The learning rate applied to every gradient step.
    """

    def __init__(self, eta=1.):

        self.eta = eta

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
        x_new = x - self.eta * grad
        return x_new

    def reset(self):
        """
        Reset the state of this updater for a new optimisation problem.
        """
        pass


class AdaDelta(SGDUpdater):
    """
    AdaDelta Algorithm

    Parameters
    ----------
    rho: float, optional
        smoothing/decay rate parameter, must be [0, 1].
    epsilon: float, optional
        "jitter" term to ensure continued learning (should be small).
    """

    def __init__(self, rho=0.1, epsilon=1e-5):

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

        x_new = x + dx
        return x_new

    def reset(self):
        """
        Reset the state of this updater for a new optimisation problem.
        """
        self.__init__(self.rho, self.epsilon)


class AdaGrad(SGDUpdater):
    """
    AdaGrad Algorithm

    Parameters
    ----------
    eta: float, optional
        smoothing/decay rate parameter, must be [0, 1].
    epsilon: float, optional
        small constant term to prevent divide-by-zeros
    """

    def __init__(self, eta=1, epsilon=1e-6):

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
        x_new = x - self.eta * grad / (self.epsilon + np.sqrt(self.g2_hist))
        return x_new

    def reset(self):
        """
        Reset the state of this updater for a new optimisation problem.
        """
        self.__init__(self.eta, self.epsilon)


class Momentum(SGDUpdater):
    """
    Momentum Algorithm

    Parameters
    ----------
    rho: float, optional
        smoothing/decay rate parameter, must be [0, 1].
    eta: float, optional
        weight to give to the momentum term
    """

    def __init__(self, rho=0.5, eta=0.01):

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

        x_new = x + self.dx
        return x_new

    def reset(self):
        """
        Reset the state of this updater for a new optimisation problem.
        """
        self.__init__(self.rho, self.eta)


class Adam(SGDUpdater):
    """
    Adam Algorithm

    Parameters
    ----------
    alpha: float, optional
        stepsize to give the update.
    beta1: float, optional
        smoothing/decay rate parameter for the gradient, must be [0, 1].
    beta2: float, optional
        smoothing/decay rate parameter for the squared gradient, must be
        [0, 1].
    epsilon: float, optional
        "jitter" term to ensure continued learning (should be small).
    """

    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def __call__(self, x, grad):
        """
        Get a new parameter value from the Adam algorithm

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
        self.t += 1

        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        mbar = self.m / (1 - self.beta1**self.t)
        vbar = self.v / (1 - self.beta2**self.t)

        x_new = x - self.alpha * mbar / (np.sqrt(vbar) + self.epsilon)
        return x_new

    def reset(self):
        """
        Reset the state of this updater for a new optimisation problem.
        """
        self.__init__(self.alpha, self.beta1, self.beta2, self.epsilon)


#
# SGD minimizer
#

def sgd(fun, x0, data, args=(), bounds=None, batch_size=10, maxiter=5000,
        updater=None, eval_obj=False, random_state=None):
    """
    Stochastic Gradient Descent.

    Parameters
    ----------
    fun: callable
        the function to evaluate, this must have the signature :code:`[obj,]
        grad = fun(x, data, ...)`, where the :code:`eval_obj` argument tells
        :code:`sgd` if an objective function value is going to be returned by
        :code:`fun`.
    x0: ndarray
        a sequence/1D array of initial values for the parameters to learn.
    data: ndarray
        a numpy array or sequence of data to input into :code:`fun`. This will
        be split along the first axis (axis=0), and then input into
        :code:`fun`.
    args: sequence, optional
        an optional sequence of arguments to give to fun.
    bounds: sequence, optional
        Bounds for variables, (min, max) pairs for each element in x, defining
        the bounds on that parameter.  Use None for one of min or max when
        there is no bound in that direction.
    batch_size: int, optional
        The number of observations in an SGD batch.
    maxiter: int, optional
        Number of mini-batch iterations before optimization terminates.
    updater: SGDUpdater, optional
        The type of gradient update to use, by default this is Adam.
    eval_obj: bool, optional
        This indicates whether or not :code:`fun` also evaluates and returns
        the objective function value. If this is true, :code:`fun` must return
        :code:`(obj, grad)` and then a list of objective function values is
        also returned.
    random_state: int or RandomState, optional
        random seed

    Returns
    -------
    res: OptimizeResult
        x: narray
            the final result
        norms: list
            the list of gradient norms
        message: str
            the convergence condition ('maxiter reached' or error)
        objs: list
            the list of objective function evaluations if :code:`eval_obj`
            is True.
        fun: float
            the final objective function evaluation if :code:`eval_obj` is
            True.
    """

    if updater is None:
        updater = Adam()

    # Make sure we aren't using a recycled updater
    updater.reset()

    N = _len_data(data)
    x = np.array(x0, copy=True, dtype=float)
    D = x.shape[0]

    # Make sure we have a valid batch size
    batch_size = min(batch_size, N)

    # Process bounds
    if bounds is not None:
        if len(bounds) != D:
            raise ValueError("The dimension of the bounds does not match x0!")

        lower, upper = zip(*map(normalize_bound, bounds))
        lower = np.array(lower)
        upper = np.array(upper)

    # Learning Records
    obj = None
    objs = []
    norms = []

    for ind in sgd_iter(maxiter, N, batch_size, random_state):

        if not eval_obj:
            grad = fun(x, *chain(_split_data(data, ind), args))
        else:
            obj, grad = fun(x, *chain(_split_data(data, ind), args))
            objs.append(obj)

        norms.append(np.linalg.norm(grad))

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
            x = np.clip(x, lower, upper)

    # Format results
    res = OptimizeResult(
        x=x,
        norms=norms,
        message='maxiter reached',
        fun=obj,
        objs=objs
    )

    return res


def sgd_iter(maxiter, N, batch_size, random_state=None):
    """
    Create random batches for Stochastic gradients.

    Batch index generator for SGD that will yeild random batches for a
    a defined number of iterations. This calls _sgd_pass until the required
    number of iterations have been reached.

    Parameters
    ----------
    maxiter: int
        The number of iterations
    N: int
        length of dataset.
    batch_size: int
        number of data points in each batch.
    random_state: int or RandomState, optional
        random seed

    Yields
    ------
    ndarray:
        of size (batch_size,) of random (int).
    """
    perms = endless_permutations(N, random_state)

    for _ in range(maxiter):
        yield np.array([next(perms) for _ in range(batch_size)])


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


#
# Module Helpers
#

def _len_data(data):

    if not issequence(data):
        return data.shape[0]

    N = len(data[0])
    for d in data[1:]:
        if d.shape[0] != N:
            raise ValueError("Not all data is the same length!")

    return N


def _split_data(data, ind):

    if not issequence(data):
        return (data[ind],)

    return [d[ind] for d in data]
