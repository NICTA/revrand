"""Bound types and bounded parameter types."""

from collections import namedtuple
from itertools import chain

import numpy as np
from sklearn.utils import check_random_state


class _BoundMixin(object):
    """Methods for the bound objects."""

    def check(self, value):
        """
        Check a value falls within a bound.

        Parameters
        ----------
        value : scalar or ndarray
            value to test

        Returns
        -------
        bool:
            If all values fall within bounds

        Example
        -------
        >>> bnd = Bound(1, 2)
        >>> bnd.check(1.5)
        True
        >>> bnd.check(3)
        False
        >>> bnd.check(np.ones(10))
        True
        >>> bnd.check(np.array([1, 3, 1.5]))
        False
        """
        if self.lower:
            if np.any(value < self.lower):
                return False

        if self.upper:
            if np.any(value > self.upper):
                return False

        return True

    def clip(self, value):
        """
        Clip a value to a bound.

        Parameters
        ----------
        value : scalar or ndarray
            value to clip

        Returns
        -------
        scalar or ndarray :
            of the same shape as value, bit with each element clipped to fall
            within the specified bounds

        Example
        -------
        >>> bnd = Bound(1, 2)
        >>> bnd.clip(1.5)
        1.5
        >>> bnd.clip(3)
        2
        >>> bnd.clip(np.array([1, 3, 1.5]))
        array([ 1. ,  2. ,  1.5])
        >>> bnd = Bound(None, None)
        >>> bnd.clip(np.array([1, 3, 1.5]))
        array([ 1. ,  3. ,  1.5])
        """
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
    A Parameter class that associates a value with a bound.

    Attributes
    ----------
    value : scalar, ndarray, scipy.stats, optional
        a value or distribution to associate with this parameter. This is
        typically used as an initial value for an optimizer, and if a random
        starts optimiser is used (eg. revrand.optimize.structured_minimizer) it
        can draw randomly from the distribution.
    bound : Bound, optional
        a Bound tuple that describes the valid range for all of the elements in
        value
    shape : tuple, optional
        the shape of value, this is ignored if value is not a scipy.stats
        distribution (i.e. it is automatically determined).

    Note
    ----
    - If ``value`` is initialised as a distribution from ``scipy.stats``, then
      ``self.value`` is actually the mean of the distribution.
    - If you want to set value to a ``scipy.stats`` distribution, and also
      associate it with a ``shape`` (i.e. you want an ``ndarray`` random
      variable), then also set the ``shape`` parameters to the desired
      dimensions.

    Examples
    --------
    Null

    >>> p = Parameter()
    >>> p.value
    []
    >>> p.has_value
    False

    Scalar

    >>> p = Parameter(1.2, Bound(1, 2))
    >>> p.shape
    ()
    >>> p.value
    1.2
    >>> p.rvs()
    1.2
    >>> p.is_random
    False

    ndarray

    >>> p = Parameter(np.ones(3), Positive())
    >>> p.shape
    (3,)
    >>> p.value
    array([ 1.,  1.,  1.])
    >>> p.rvs()
    array([ 1.,  1.,  1.])

    ``scipy.stats`` scalar

    >>> from scipy.stats import gamma
    >>> p = Parameter(gamma(a=1, scale=1), Positive())
    >>> p.value == gamma(a=1, scale=1).mean()
    True
    >>> np.isscalar(p.rvs())
    True
    >>> p.is_random
    True

    ``scipy.stats`` ndarray

    >>> from scipy.stats import gamma
    >>> p = Parameter(gamma(a=1, scale=1), Positive(), shape=(3,))
    >>> all(p.value == np.ones(3) * gamma(a=1, scale=1).mean())
    True
    >>> p.rvs().shape == (3,)
    True
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
        r"""
        Draw a random value from this Parameter's distribution.

        If ``value`` was not initialised with a ``scipy.stats`` object, then
        the scalar/ndarray value is returned.

        Parameters
        ----------
        random_state : None, int or RandomState, optional
            random seed

        Returns
        -------
        ndarray :
            of size ``self.shape``, a random draw from the distribution, or
            ``self.value`` if not initialised with a ``scipy.stats`` object.

        Note
        ----
        Random draws are *clipped* to the bounds, and so it is up to the user
        to input a sensible sampling distribution!
        """
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
        """Test if this Parameter has a value, or is "null"."""
        return self.shape != (0,)

    @property
    def is_random(self):
        """Test if this Parameter was initialised with a distribution."""
        return self.dist is not None


def ravel(parameter, random_state=None):
    """
    Flatten a ``Parameter``.

    Parameters
    ----------
    parameter: Parameter
        A ``Parameter`` object

    Returns
    -------
    flatvalue: ndarray
        a flattened array of shape ``(prod(parameter.shape),)``
    flatbounds: list
        a list of bound tuples of length ``prod(parameter.shape)``
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
        a sequence of value, ``Bound`` pairs

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
    Get the shape of a ``Parameter``.

    Parameters
    ----------
    parameter: Parameter
        ``Parameter`` object to get the shape of

    Returns
    -------
    tuple:
        shape of the ``Parameter`` object
    """
    return parameter.shape
