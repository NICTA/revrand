"""Bound types and bounded parameter types."""

from collections import namedtuple
from itertools import chain

import numpy as np
from sklearn.utils import check_random_state


class _BoundMixin(object):

    def check(self, value):

        if self.lower:
            if np.any(value < self.lower):
                return False

        if self.upper:
            if np.any(value > self.upper):
                return False

        return True

    def clip(self, value):

        if not self.lower and not self.upper:
            return value

        return np.clip(value, self.lower, self.upper)


class Bound(namedtuple('Bound', ['lower', 'upper']), _BoundMixin):
    """
    Define bounds on a variable for the optimiser.

    This defaults to all real values allowed (i.e. no bounds).

    Parameters
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.
    dist : scipy.stats.distibution, optional
        a distibution object that specifies how to sample values from this
        bound

    Attributes
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.
    dist : scipy.stats.distibution, optional
        a distibution object that specifies how to sample values from this
        bound

    Examples
    --------
    >>> b = Bound(1e-10, upper=1e-5)
    >>> b
    Bound(lower=1e-10, upper=1e-05)
    >>> b.lower
    1e-10
    >>> b.upper
    1e-05
    >>> isinstance(b, tuple)
    True
    >>> tuple(b)
    (1e-10, 1e-05)
    >>> lower, upper = b
    >>> lower
    1e-10
    >>> upper
    1e-05
    >>> Bound(42, 10)
    Traceback (most recent call last):
        ...
    ValueError: lower bound cannot be greater than upper bound!
    """

    def __new__(cls, lower=None, upper=None):

        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError('lower bound cannot be greater than upper '
                                 'bound!')
        return super(Bound, cls).__new__(cls, lower, upper)

    def __getnewargs__(self):
        """Required for pickling."""
        return (self.lower, self.upper)


class Positive(namedtuple('Positive', ['lower', 'upper']), _BoundMixin):
    """
    Define a positive only bound for the optimiser.

    This may induce the 'log trick' in the optimiser (when using an appropriate
    decorator), which will ignore the 'smallest' value (but will stay above 0).

    Parameters
    ---------
    upper : float
        The largest value allowed for the optimiser to evaluate (if not using
        the log trick).
    dist : scipy.stats.distibution, optional
        a distibution object that specifies how to sample values from this
        bound

    Examples
    --------
    >>> b = Positive()
    >>> b
    Positive(lower=1e-14, upper=None)

    Since ``tuple`` (and by extension its descendents) are immutable,
    the lower bound for all instances of ``Positive`` are guaranteed to
    be positive.
    """

    def __new__(cls, upper=None):

        lower = 1e-14
        if upper is not None:
            if lower > upper:
                raise ValueError('Upper bound must be greater than {}'
                                 .format(lower))

        return super(Positive, cls).__new__(cls, lower, upper)

    def __getnewargs__(self):
        """Required for pickling."""
        return (self.upper,)


class Parameter(object):
    """
    A Parameter class that associates a value (scalar or ndarray) with a bound.

    Attributes
    ----------
    value: scalar or ndarray, optional
        a value to associate with this parameter. This is typically used as an
        initial value for an optimizer.
    bound: Bound
        a Bound tuple that describes the valid range for all of the elements in
        value
    shape: tuple
        the shape of value
    """

    def __init__(self, value=[], bounds=Bound(), shape=()):

        if not hasattr(value, 'rvs'):
            if np.any(value) and not bounds.check(value):
                raise ValueError("Value not within bounds!")

            self.value = value
            self.dist = None
            self.shape = np.shape(value)
        else:
            self.dist = value
            self.shape = shape
            mean = bounds.clip(self.dist.mean())
            self.value = mean if shape == () else mean * np.ones(shape)

        self.bounds = bounds

    def rvs(self, random_state=None):

        # No sampling distibution
        if self.dist is None:
            return self.value

        # Unconstrained samples
        rs = check_random_state(random_state)
        samples = self.dist.rvs(size=self.shape, random_state=rs)

        # Bound the samples
        samples = self.bounds.clip(samples)

        return samples

    @property
    def has_value(self):
        return self.shape != (0,)

    @property
    def is_random(self):
        return self.dist is not None


def ravel(parameter, random_state=None):
    """
    Flatten a :code:`Parameter`.

    Parameters
    ----------
    parameter: Parameter
        A :code:`Parameter` object

    Returns
    -------
    flatvalue: ndarray
        a flattened array of shape :code:`(prod(parameter.shape),)`
    flatbounds: list
        a list of bound tuples of length :code:`prod(parameter.shape)`
    """
    flatvalue = np.ravel(parameter.rvs(random_state=random_state))
    flatbounds = [parameter.bounds
                  for _ in range(np.prod(parameter.shape, dtype=int))]

    return flatvalue, flatbounds


def hstack(tup):
    """
    Horizontally stack a sequence of value bounds pairs.

    Parameters
    ----------
    tup: sequence
        a sequence of value, :code:`Bound` pairs

    Returns
    -------
    value: ndarray
        a horizontally concatenated array1d
    bounds:
        a list of Bounds
    """
    vals, bounds = zip(*tup)
    stackvalue = np.hstack(vals)
    stackbounds = list(chain(*bounds))

    return stackvalue, stackbounds


def shape(parameter):
    """
    Get the shape of a :code:`Parameter`.

    Parameters
    ----------
    parameter: Parameter
        :code:`Parameter` object to get the shape of

    Returns
    -------
    tuple:
        shape of the :code:`Parameter` object
    """
    return parameter.shape
