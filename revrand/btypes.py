import numpy as np
from collections import namedtuple


class Bound(namedtuple('Bound', ['lower', 'upper'])):
    """
    Define bounds on a variable for the optimiser. This defaults to all
    real values allowed (i.e. no bounds).

    Parameters
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.

    Attributes
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.

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
        obj = super(Bound, cls).__new__(cls, lower, upper)

        return obj

    # TODO: Transformation details for optimiser (logistic/identity)


class Positive(Bound):
    """
    Define a positive only bound for the optimiser. This may induce the
    'log trick' in the optimiser (when using an appropriate decorator), which
    will ignore the 'smallest' value (but will stay above 0).

    Parameters
    ---------
    upper : float
        The largest value allowed for the optimiser to evaluate (if not using
        the log trick).

    Examples
    --------
    >>> b = Positive()
    >>> b # doctest: +SKIP
    Positive(lower=1e-14, upper=None)

    Since ``tuple`` (and by extension its descendents) are immutable,
    the lower bound for all instances of ``Positive`` are guaranteed to
    be positive.
    """
    def __new__(cls, upper=None):

        return super(Positive, cls).__new__(cls, lower=1e-14, upper=upper)

    def __getnewargs__(self):
        """Required for pickling!"""
        return (self.upper,)

    # TODO: Transformation details for optimiser (log)


class Parameter(object):

    def __init__(self, value, bounds=Bound()):

        self.value = value
        self.shape = (1,) if np.isscalar(value) else value.shape
        self.bounds = bounds


def get_values(parameters):

    if isinstance(parameters, Parameter):
        return parameters.value

    return [p.value for p in parameters]


def flatten_bounds(parameters):

    inflate = lambda p: [p.bounds for _ in range(np.prod(p.shape))]

    if isinstance(parameters, Parameter):
        return inflate(parameters)

    return [b for p in parameters for b in inflate(p)]
