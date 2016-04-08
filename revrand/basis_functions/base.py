""" Various basis function objects specialised for parameter learning.

    To make a new basis object, see the documentation of the Basis class.
"""

from __future__ import division

import sys
import inspect
import numpy as np
from functools import wraps
from scipy.linalg import norm
from scipy.special import gammaincinv, expit
from scipy.spatial.distance import cdist

from ..optimize import Positive
from ..linalg import hadamard

# TODO:
# - Remove the need to bases to know their params bounds, see #54 and #55
# - Implement basis function input slicing, see #53


#
# Module Helper Functions
#

# For basis concatenation functionality
if sys.version_info[0] < 3:
    def count_args(func):
        nargs = len(inspect.getargspec(func)[0])
        return nargs - 1 if inspect.ismethod(func) else nargs  # remove self
else:
    def count_args(func):
        return len((inspect.signature(func)).parameters)


# For basis function slicing
def slice_init(func):

    @wraps(func)
    def new_init(self, *args, **kwargs):

        apply_ind = kwargs.pop('apply_ind', None)
        if np.isscalar(apply_ind):
            apply_ind = [apply_ind]

        func(self, *args, **kwargs)
        self.apply_ind = apply_ind

    return new_init


def slice_call(func):

    @wraps(func)
    def new_call(self, X, *args, **kwargs):

        X = X if self.apply_ind is None else X[:, self.apply_ind]
        return func(self, X, *args, **kwargs)

    return new_call


# Calculating function gradients w.r.t. structured basis functions
def apply_grad(fun, grad):

    if inspect.isgenerator(grad):
        fgrad = [apply_grad(fun, g) for g in grad]
        return fgrad if len(fgrad) != 1 else fgrad[0]
    elif len(grad) == 0:
        return []
    elif grad.ndim == 2:
        return fun(grad)
    elif grad.ndim == 3:
        return np.array([fun(grad[:, :, i]) for i in range(grad.shape[2])])
    else:
        raise ValueError("Only 2d or 3d gradients allowed!")


#
# Basis objects
#

class Basis(object):
    """ The base Basis class. To make other basis classes, make sure they are
        subclasses of this class to enable concatenation and operation with the
        machine learning algorithms.

        Example:
            Basis concatentation works as follows if you subclass this class:

                ConcatBasis = MyBasis1(properties1) + MyBasis2(properties2)
    """

    _bounds = []

    @slice_init
    def __init__(self):
        """
        Construct this an instance of this class. This is also a good place
        to set non-learnable properties, and bounds on the parameters. An
        example Basis class with parameters may be,

        Example:

        .. code-block:: python

            def __init__(self, property, param_bounds=(1e-7, None)):

                self.property = property
                self.bounds = [params_bounds]

        All basis class objects MUST have a bounds property, which is either:

        -   an empty list
        -   a list of pairs of upper and lower bounds for each parameter.
            This is a concatenated list over all parameters, including
            vector parameters! See the minimize module for a guide on how
            these bounds work.
        """
        pass

    @slice_call
    def __call__(self, X):
        """ Return the basis function applied to X, i.e. Phi(X, params), where
            params can also optionally be used and learned.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.
                params: optional parameter aguments, these can be scalars or
                    arrays.

            Returns:
                array: of shape (N, D) where D is the number of basis
                    functions.
        """
        return X

    @slice_call
    def grad(self, X):
        """ Return the gradient of the basis function w.r.t.\ each of the
            parameters.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.
                params: optional parameter aguments, these can be scalars or
                    arrays.

            Returns:
                list: with each element being an (N, D) array (same
                    dimensionality as return by __call__()) of a gradient with
                    respect to a parameter. The length of this list must be the
                    same as the *total* number of scalars in all of the
                    parameters, i.e. the same length as Basis.bounds.

                    The exception to this is if there are *no* parameters, in
                    which case a list of one element, containing an array of
                    (N, D) zeros must be returned.
            """
        return []

    def _call_popargs(self, X, *args):

        selfargs, otherargs = self._splitargs(args, self.__call__, offset=1)

        return self.__call__(X, *selfargs), otherargs

    def _grad_popargs(self, X, *args):

        selfargs, otherargs = self._splitargs(args, self.grad, offset=1)

        return self.grad(X, *selfargs), otherargs, selfargs

    def _splitargs(self, args, fn, offset=0):

        nargs = count_args(fn) - offset
        selfargs, otherargs = args[:nargs], args[nargs:]

        return selfargs, otherargs

    @property
    def bounds(self):
        """ Get this objects parameter bounds. This is a list of pairs of upper
            and lower bounds, with the same length as the total number of
            scalars in all of the parameters combined (and in order).
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """ Set this objects parameter bounds. This is a list of pairs of upper
            and lower bounds, with the same length as the total number of
            scalars in all of the parameters combined (and in order).
        """
        self._bounds = bounds

    def __add__(self, other):

        return BasisCat([self, other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)


class LinearBasis(Basis):
    """ Linear basis class, basically this just prepending a columns of ones
        onto X.
    """

    @slice_init
    def __init__(self, onescol=False):
        """ Construct a linear basis object.

            Arguments:
                onescol: If true, prepend a column of ones onto X.
        """

        self.onescol = onescol

    @slice_call
    def __call__(self, X):
        """ Return this basis applied to X.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.

            Returns:
                array: of shape (N, d+1), or (N, d) depending on onescol.
        """

        N, D = X.shape
        return np.hstack((np.ones((N, 1)), X)) if self.onescol else X


class PolynomialBasis(Basis):
    """ Polynomial basis class, this essentially creates the concatenation,
        Phi = [X^0, X^1, ..., X^p] where p is specified in the constructor.
    """

    @slice_init
    def __init__(self, order, include_bias=True):
        """ Construct a polynomial basis object.

            Arguments:
                order: the order of the polynomial to create, i.e. the last
                    power to raise X to in the concatenation Phi = [X^0, X^1,
                    ..., X^order].
                include_bias: If True (default), include the bias column
                    (column of ones which acts as the intercept term in a
                    linear model)
        """

        if order < 0:
            raise ValueError("Polynomial order must be positive")
        self.order = order

        self.include_bias = include_bias

    @slice_call
    def __call__(self, X):
        """ Return this basis applied to X.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.

            Returns:
                array: of shape (N, d*order+1), the extra 1 is from a
                    prepended ones column.
        """

        N, D = X.shape

        pow_arr = np.arange(self.order) + 1

        # Polynomial terms
        Phi = X[:, :, np.newaxis] ** pow_arr

        # Flatten along last axes
        Phi = Phi.reshape(N, D * self.order)

        # Prepend intercept
        if self.include_bias:
            Phi = np.hstack((np.ones((N, 1)), Phi))

        # TODO: Using np.hstack is about 4x slower than initializing, say,
        # an N by d*order+1 ndarray of ones and assigning the remaining
        # N by d*order values. May want to revisit this implementation.

        return Phi


class RadialBasis(Basis):
    """
    Radial basis class.

    Note:
        This will have relevance vector machine-like behaviour with
        uncertainty and for deaggregation tasks!
    """

    @slice_init
    def __init__(self, centres, lenscale_bounds=Positive()):
        """
        Construct a radial basis function (RBF) object.

        Arguments:
            centres:    array of shape (Dxd) where D is the number of centres
                        for the radial bases, and d is the dimensionality of X.

            lenscale_bounds: a tuple of bounds for the RBFs' length scales.
        """

        self.M, self.D = centres.shape
        self.C = centres
        self.bounds = lenscale_bounds

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the RBF to X.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            array:  of shape (N, D) where D is number of RBF centres.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        return np.exp(- cdist(X, self.C, 'sqeuclidean') / (2 * lenscale**2))

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scale.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            list:   with one element of shape (N, D) where D is number of RBF
                    centres. This is d Phi(X) / d lenscale.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        sdist = cdist(X, self.C, 'sqeuclidean')
        dPhi = np.exp(- sdist / (2 * lenscale**2)) * sdist / lenscale**3

        return dPhi


# TODO: Might be worth creating a mixin or base class for basis functions
# that require locations and scales

class SigmoidalBasis(Basis):
    """Sigmoidal Basis"""

    @slice_init
    def __init__(self, centres, lenscale_bounds=Positive()):
        """Construct a sigmoidal basis function object.

        Arguments:
            centres: array of shape (Dxd) where D is the number of centres
                for the_call_poparg bases, and d is the dimensionality of X.
            lenscale_bounds: a tuple of bounds for the basis function length
                scales.
        """

        self.M, self.D = centres.shape
        self.C = centres
        self.bounds = lenscale_bounds

    @slice_call
    def __call__(self, X, lenscale):
        r"""Apply the sigmoid basis function to X.

        .. math::

            \phi_j (x) = \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )

        where :math:`\sigma` is the logistic sigmoid function defined by

        .. math::

            \sigma(a) = \frac{1}{1+e^{-a}}

        Arguments:
            X: (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.
            lenscale: the length scale (scalar) of the basis functions to
                apply to X.

        Returns:
            array: of shape (N, D) where D is number of centres.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("Expected X of dimensionality {0}, got {1}"
                             .format(self.D, D))

        return expit(cdist(X, self.C, 'seuclidean') / lenscale)

    @slice_call
    def grad(self, X, lenscale):
        r"""Get the gradients of this basis w.r.t.\ the length scale.

        .. math::

            \frac{\partial}{\partial s} \phi_j(x) =
            - \frac{\| x - \mu_j \|_2}{s^2}
            \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )
            \left ( 1 - \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )
            \right )

        Arguments:
            X: (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the  to apply to X.

        Returns:
            list: with one element of shape (N, D) where D is number of
                centres. This is d Phi(X) / d lenscale.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("Expected X of dimensionality {0}, got {1}"
                             .format(self.D, D))

        dist = cdist(X, self.C, 'seuclidean')

        sigma = expit(dist / lenscale)

        dPhi = - dist * sigma * (1 - sigma) / lenscale**2

        return dPhi


class RandomRBF(RadialBasis):
    """
    Random RBF Basis, otherwise known as Random Kitchen Sinks.

    This will make a linear regression model approximate a GP with an RBF
    covariance function.
    """

    @slice_init
    def __init__(self, nbases, Xdim, lenscale_bounds=Positive()):
        """
        Construct a random radial basis function (RBF) object.

        Arguments:
            nbases: a scalar for how many random bases to create.

            Xdim: the dimension (d) of the observations.

            lenscale_bounds: a tuple of bounds for the RBFs' length scales.
        """
        self.d = Xdim
        self.n = nbases
        self.W = np.random.randn(self.d, self.n)
        self.bounds = lenscale_bounds

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the random RBF to X.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            array:  of shape (N, 2*nbases) where nbases is number of random
                    bases to use, given in the constructor.
        """

        N, D = X.shape
        self._checkD(D)

        WX = np.dot(X, self.W / lenscale)

        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(self.n)

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scale.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            list:   with one element of shape (N, 2*nbases) where nbases is
                    number of random RBF bases. This is d Phi(X) / d lenscale.
        """

        N, D = X.shape
        self._checkD(D)

        WX = np.dot(X, self.W / lenscale)
        dWX = WX / lenscale

        return np.hstack((dWX * np.sin(WX), -dWX * np.cos(WX))) \
            / np.sqrt(self.n)

    def _checkD(self, D):
        if D != self.d:
            raise ValueError("Dimensions of data inconsistent!")


class RandomRBF_ARD(RandomRBF):
    """ Random RBF Basis, otherwise known as Random Kitchen Sinks, with
        automatic relevance determination (ARD).

        This will make a linear regression model approximate a GP with an
        ARD-RBF covariance function.
    """

    @slice_init
    def __init__(self, nbases, Xdim, lenscale_bounds=Positive()):
        """ Construct a random radial basis function (RBF) object, with ARD.

            Arguments:
                nbases: a scalar for how many random bases to create.
                Xdim: the dimension (d) of the observations.
                lenscale_bounds: a tuple of bounds for the RBFs' length scales.
        """

        # lenscale_bounds.shape = Xdim
        super(RandomRBF_ARD, self).__init__(nbases, Xdim, lenscale_bounds)
        lenscale_bounds.shape = Xdim
        self.bounds = lenscale_bounds

    @slice_call
    def __call__(self, X, lenscales):
        """ Apply the random ARD-RBF to X.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.
                lenscale: array of shape (d,) length scales (one for each
                    dimension of X).

            Returns:
                array: of shape (N, 2*nbases) where nbases is number of random
                    bases to use, given in the constructor.
        """

        N, D = X.shape
        self._checkD(D, len(lenscales))

        WX = np.dot(X, self.W / np.asarray(lenscales)[:, np.newaxis])

        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(self.n)

    @slice_call
    def grad(self, X, lenscales):
        """ Get the gradients of this basis w.r.t.\ the length scales.

            Arguments:
                X: (N, d) array of observations where N is the number of
                    samples, and d is the dimensionality of X.
                lenscale: array of shape (d,) length scales (one for each
                    dimension of X).

            Returns:
                list: with d arrays of shape (N, 2*nbases) where nbases is
                    number of random RBF bases. This is d Phi(X) / d lenscale
                    for each length scale parameter.
        """

        N, D = X.shape
        self._checkD(D, len(lenscales))

        WX = np.dot(X, self.W / np.asarray(lenscales)[:, np.newaxis])
        sinWX = - np.sin(WX)
        cosWX = np.cos(WX)

        dPhi = []
        for i, l in enumerate(lenscales):
            dWX = np.outer(X[:, i], - 1. / l**2 * self.W[i, :])
            dPhi.append(np.hstack((dWX * sinWX, dWX * cosWX))
                        / np.sqrt(self.n))

        return np.dstack(dPhi)

    def _checkD(self, Xdim, lendim):
        if Xdim != self.d:
            raise ValueError("Dimensions of data inconsistent!")
        if lendim != self.d:
            raise ValueError("Dimensions of lenscale inconsistent!")


class FastFood(RandomRBF):
    """
    Fast Food basis function, which is an approximation of the random
    radial basis function for a large number of bases.

    This will make a linear regression model approximate a GP with an RBF
    covariance function.
    """

    @slice_init
    def __init__(self, nbases, Xdim, lenscale_bounds=Positive()):
        """
        Construct a random radial basis function (RBF) object.

        Arguments:
            nbases: a scalar for how many random bases to create
                    approximately, this actually will be to the neareset larger
                    two power.

            Xdim:   the dimension (d) of the observations.
                    lenscale_bounds: a tuple of bounds for the RBFs' length
                    scales.
        """

        self.bounds = lenscale_bounds

        # Make sure our dimensions are powers of 2
        l = int(np.ceil(np.log2(Xdim)))

        self.d = Xdim
        self.d2 = pow(2, l)
        self.k = int(np.ceil(nbases / self.d2))
        self.n = self.d2 * self.k

        # Draw consistent samples from the covariance matrix
        results = [self.__sample_params() for i in range(self.k)]
        self.B, self.G, self.PI, self.S = tuple(zip(*results))

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the Fast Food RBF basis to X.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            array:  of shape (N, 2*nbases) where nbases is number of random
                    bases to use, given in the constructor (to nearest larger
                    two power).
        """

        self._checkD(X.shape[1])

        VX = self.__makeVX(X) / lenscale
        Phi = np.hstack((np.cos(VX), np.sin(VX))) / np.sqrt(self.n)
        return Phi

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scale.

        Arguments:
            X:  (N, d) array of observations where N is the number of
                samples, and d is the dimensionality of X.

            lenscale: the length scale (scalar) of the RBFs to apply to X.

        Returns:
            list:   with one element of shape (N, 2*nbases) where nbases is
                    number of random RBF bases, again to the nearest larger
                    two power. This is d Phi(X) / d lenscale.
        """

        self._checkD(X.shape[1])

        VX = self.__makeVX(X) / lenscale
        dVX = - VX / lenscale

        return np.hstack((-dVX * np.sin(VX), dVX * np.cos(VX))) \
            / np.sqrt(self.n)

    def __sample_params(self):

        B = np.random.randint(2, size=self.d2) * 2 - 1  # uniform from [-1,1]
        G = np.random.randn(self.d2)  # mean 0 std 1
        PI = np.random.permutation(self.d2)
        S = np.sqrt(2 * gammaincinv(np.ceil(self.d2 / 2),
                                    np.random.rand(self.d2))) / norm(G)
        return B, G, PI, S

    def __makeVX(self, X):
        m, d0 = X.shape

        # Pad the dimensions of X to nearest 2 power
        X_dash = np.zeros((m, self.d2))
        X_dash[:, 0:d0] = X

        VX = []
        for B, G, PI, S in zip(*(self.B, self.G, self.PI, self.S)):
            vX = hadamard(X_dash * B[np.newaxis, :], ordering=False)
            vX = vX[:, PI] * G[np.newaxis, :]
            VX.append(hadamard(vX, ordering=False) * S[np.newaxis, :]
                      * np.sqrt(self.d2))

        return np.hstack(VX)


#
# Other basis construction objects and functions
#

class BasisCat(object):
    """ A class that implements concatenation of bases. """

    def __init__(self, basis_list):

        self.bases = basis_list

    def __call__(self, X, *params):

        Phi = []
        args = params

        for base in self.bases:
            phi, args = base._call_popargs(X, *args)
            Phi.append(phi)

        return np.hstack(Phi)

    def grad(self, X, *params):

        # Get all gradients
        N = X.shape[0]
        args = list(params)
        grads = []
        dims = [0]

        for base in self.bases:
            g, args, sargs = base._grad_popargs(X, *args)
            grads.append(g)
            dims.append(base(X, *sargs).shape[1] if len(g) == 0 else
                        g.shape[1])

        # Now generate structured gradients
        D = np.sum(dims)
        endinds = np.cumsum(dims)

        for i, g in enumerate(grads):

            if len(g) == 0:
                continue

            dPhi_dim = (N, D) if g.ndim < 3 else (N, D, g.shape[2])
            dPhi = np.zeros(dPhi_dim)
            dPhi[:, endinds[i]:endinds[i + 1]] = g

            yield dPhi

    @property
    def bounds(self):

        bounds = [b.bounds for b in self.bases if len(b.bounds) > 0]

        return bounds[0] if len(bounds) == 1 else bounds

    def __add__(self, other):

        if isinstance(other, BasisCat):
            return BasisCat(self.bases + other.bases)
        else:
            return BasisCat(self.bases + [other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)
