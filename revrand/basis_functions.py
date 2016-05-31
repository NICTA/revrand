""" Various basis function objects specialised for parameter learning.

    To make a new basis object, see the documentation of the Basis class.
"""

from __future__ import division

import sys
import inspect
import numpy as np
from six import wraps
from decorator import decorator  # Preserves function signature (pyth2 compat)
from scipy.linalg import norm
from scipy.special import gammaincinv, expit
from scipy.spatial.distance import cdist
from scipy.stats import cauchy, laplace, t

from .btypes import Positive, Parameter
from .math.linalg import hadamard
from .math.special import safediv


#
# Module Helper Functions
#

# For basis concatenation functionality
if sys.version_info[0] < 3:
    def count_args(func):
        """
        Count the number of arguments in a function/method.

        Parameters
        ----------
        func: callable
            a function or class method

        Returns
        -------
        int:
            the number of arguments, excluding self
        """
        nargs = len(inspect.getargspec(func)[0])
        return nargs - 1 if inspect.ismethod(func) else nargs  # remove self
else:
    def count_args(func):
        """
        Count the number of arguments in a function/method.

        Parameters
        ----------
        func: callable
            a function or class method

        Returns
        -------
        int:
            the number of arguments, excluding self
        """
        return len((inspect.signature(func)).parameters)


# For basis function slicing

def slice_init(func):
    """
    Decorator for adding partial application functionality to a basis object.

    This will add an "apply_ind" argument to a basis object initialiser that
    can be used to apply the basis function to only the dimensions specified in
    apply_ind. E.g.,

    >>> X = np.ones((100, 20))
    >>> base = LinearBasis(onescol=False, apply_ind=slice(0, 10))
    >>> base(X).shape
    (100, 10)
    """

    @wraps(func)
    def new_init(self, *args, **kwargs):

        apply_ind = kwargs.pop('apply_ind', None)
        if np.isscalar(apply_ind):
            apply_ind = [apply_ind]

        func(self, *args, **kwargs)
        self.apply_ind = apply_ind

    return new_init


@decorator  # This needs to be signature preserving for concatenation
def slice_call(func, self, X, *vargs, **kwargs):
    """
    Decorator for implementing partial application.

    This must decorate the :code:`__call__` and :code:`grad` methods of basis
    objects if the :code:`slice_init` decorator was used.
    """

    X = X if self.apply_ind is None else X[:, self.apply_ind]
    return func(self, X, *vargs, **kwargs)


# Calculating function gradients w.r.t. structured basis functions
def apply_grad(fun, grad):
    """
    Apply a function that takes a gradient matrix to a sequence of 2 or 3
    dimensional gradients.

    This is partucularly useful when the gradient of a basis concatenation
    object is quite complex, eg.

    >>> X = np.random.randn(100, 3)
    >>> y = np.random.randn(100)
    >>> N, d = X.shape
    >>> base = RandomRBF(Xdim=d, nbases=5) + RandomRBF(Xdim=d, nbases=5,
    ... lenscale_init=Parameter(np.ones(d), Positive()))
    >>> Phi = base(X, 1., np.ones(d))
    >>> dffun = lambda dPhi: y.dot(Phi).dot(dPhi.T).dot(y)
    >>> df = apply_grad(dffun, base.grad(X, 1., np.ones(d)))
    >>> np.isscalar(df[0])
    True
    >>> df[1].shape
    (3,)

    Parameters
    ----------
    fun: callable
        the function too apply to the (2d) gradient.
    grad: ndarray or generator
        the gradient of the basis function (output of base.grad).

    Returns
    -------
    scalar, ndarray or sequence:
        the result of applying fun(grad) for a structured grad.
    """

    if inspect.isgenerator(grad):
        fgrad = [apply_grad(fun, g) for g in grad]
        return fgrad if len(fgrad) != 1 else fgrad[0]
    elif len(grad) == 0:
        return []
    elif (grad.ndim == 1) or (grad.ndim == 2):
        return fun(grad)
    elif grad.ndim == 3:
        return np.array([fun(grad[:, :, i]) for i in range(grad.shape[2])])
    else:
        raise ValueError("Only up to 3d gradients allowed!")


#
# Basis objects
#

class Basis(object):
    """ 
    The base Basis class. To make other basis classes, make sure they are
    subclasses of this class to enable concatenation and operation with the
    machine learning algorithms.

    Example
    -------
    Basis concatentation works as follows if you subclass this class:

    >>> base = MyBasis1(properties1) + MyBasis2(properties2)  # doctest: +SKIP
    """

    _params = []

    @slice_init
    def __init__(self):
        """
        Construct this an instance of this class. This is also a good place
        to set non-learnable properties, and bounded Parameter types. An
        example Basis class with parameters may be,

        Example:

        .. code-block:: python

            def __init__(self, property, param_init=Parameter(1, Bound())):

                self.property = property
                self.params = param_init

        All the :code:`params` property does is inform algorithms of the
        intitial value and any bounds this basis object has. This will need to
        correspond to any parameters input into the :code:`__call__` and
        :code:`grad` methods. All basis class objects MUST have a 
        :code:`params` property, which is either:

        - an empty list for basis functions with no learnable parameters
          (just subclass this class)
        - one Parameter object for an optimisable parameter, see btypes.py 
        - a list of Parameter objects, one for each optimisable parameter
        """
        pass

    @slice_call
    def __call__(self, X):
        """ 
        Return the basis function applied to X, i.e. Phi(X, params), where
        params can also optionally be used and learned.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and 
            d is the dimensionality of X.
        params: optional
            parameter aguments, these can be scalars or arrays.

        Returns
        -------
        ndarray: 
            of shape (N, D) where D is the number of basis functions.
        """
        return X

    @slice_call
    def grad(self, X):
        """
        Return the gradient of the basis function w.r.t.\ each of the
        parameters.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        params: optional
            parameter aguments, these can be scalars or arrays.

        Returns
        -------
        list or ndarray:
            this will be a list of ndarrays if there are multiple parameters, 
            or just an ndarray if there is a single parameter. The ndarrays can
            have more than two dimensions (i.e. tensors of rank > 2), depending
            on the dimensions of the basis function parameters. If there are 
            *no* parameters, :code:`[]` is returned.
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
    def params(self):
        """ Get this object's Parameter types. """
        return self._params

    @params.setter
    def params(self, params):
        """ Set this object's Parameter types. """
        self._params = params

    def __add__(self, other):

        return BasisCat([self, other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)


class LinearBasis(Basis):
    """ 
    Linear basis class, basically this just prepends a columns of ones onto X

    Parameters
    ----------
    onescol: bool, optional
        If true, prepend a column of ones onto X.
    """

    @slice_init
    def __init__(self, onescol=False):

        self.onescol = onescol

    @slice_call
    def __call__(self, X):
        """ 
        Return this basis applied to X.

        Parameters
        ----------
        X: ndarray
            of shape (N, d) of observations where N is the number of samples, 
            and d is the dimensionality of X.

        Returns
        -------
        ndarray: 
            of shape (N, d+1), or (N, d) depending on onescol.
        """

        N, D = X.shape
        return np.hstack((np.ones((N, 1)), X)) if self.onescol else X


class PolynomialBasis(Basis):
    """
    Polynomial basis class, this essentially creates the concatenation,
    :math:`\\boldsymbol\Phi = [\mathbf{X}^0, \mathbf{X}^1, ..., \mathbf{X}^p]`
    where :math:`p` is the :code:`order` of the polynomial.

    Parameters
    ----------
    order: int
        the order of the polynomial to create, i.e. the last power to raise
        X to in the concatenation Phi = [X^0, X^1, ..., X^order].
    include_bias: bool, optional
        If True (default), include the bias column (column of ones which
        acts as the intercept term in a linear model)
    """

    @slice_init
    def __init__(self, order, include_bias=True):

        if order < 0:
            raise ValueError("Polynomial order must be positive")
        self.order = order

        self.include_bias = include_bias

    @slice_call
    def __call__(self, X):
        """
        Return this basis applied to X.

        Parameters
        ----------
        X: ndarray
            of shape (N, d) of observations where N is the number of samples, 
            and d is the dimensionality of X.

        Returns
        -------
        ndarray:
            of shape (N, d*order+1), the extra 1 is from a prepended ones 
            column.
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

    Parameters
    ----------
    centres: ndarray 
        array of shape (Dxd) where D is the number of centres for the
        radial bases, and d is the dimensionality of X.
    lenscale_init: Parameter, optional
        A scalar parameter to bound and initialise the length scales for
        optimization

    Note
    ----
    This will have relevance vector machine-like behaviour with uncertainty.
    """

    @slice_init
    def __init__(self, centres, lenscale_init=Parameter(1., Positive())):

        self.M, self.D = centres.shape
        self.C = centres
        self.params = lenscale_init

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the RBF to X.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            of shape (N, D) where D is number of RBF centres.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        return np.exp(- safediv(cdist(X, self.C, 'sqeuclidean'),
                                (2 * lenscale**2)))

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scale.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            of shape (N, D) where D is number of RBF centres. This is 
            :math:`\partial \Phi(\mathbf{X}) / \partial l`
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        sdist = cdist(X, self.C, 'sqeuclidean')
        dPhi = np.exp(- safediv(sdist, 2 * lenscale**2)) \
            * safediv(sdist, lenscale**3)

        return dPhi


class SigmoidalBasis(RadialBasis):
    """
    Sigmoidal Basis

    Parameters
    ----------
    centres: ndarray
        array of shape (Dxd) where D is the number of centres for 
        the bases, and d is the dimensionality of X.
    lenscale_init: Parameter, optional
        A scalar parameter to bound and initialise the length scales for
        optimization
    """

    @slice_call
    def __call__(self, X, lenscale):
        r"""
        Apply the sigmoid basis function to X.

        .. math::

            \phi_j (x) = \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )

        where :math:`\sigma` is the logistic sigmoid function defined by

        .. math::

            \sigma(a) = \frac{1}{1+e^{-a}}

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            of shape (N, D) where D is number of centres.
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("Expected X of dimensionality {0}, got {1}"
                             .format(self.D, D))

        return expit(safediv(cdist(X, self.C, 'euclidean'), lenscale))

    @slice_call
    def grad(self, X, lenscale):
        r"""
        Get the gradients of this basis w.r.t.\ the length scale.

        .. math::

            \frac{\partial}{\partial s} \phi_j(x) =
            - \frac{\| x - \mu_j \|_2}{s^2}
            \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )
            \left ( 1 - \sigma \left ( \frac{\| x - \mu_j \|_2}{s} \right )
            \right )

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            of shape (N, D) where D is number of centres. This is
            :math:`\partial \Phi(\mathbf{X}) / \partial l`
        """

        N, D = X.shape
        if self.D != D:
            raise ValueError("Expected X of dimensionality {0}, got {1}"
                             .format(self.D, D))

        dist = cdist(X, self.C, 'euclidean')

        sigma = expit(safediv(dist, lenscale))

        dPhi = - dist * sigma * safediv((1 - sigma), lenscale**2)

        return dPhi


class RandomRBF(Basis):
    """ 
    Random RBF Basis -- Approximates an RBF kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) RBF covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and 
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    """

    @slice_init
    def __init__(self, nbases, Xdim, lenscale_init=Parameter(1., Positive())):

        self.d = Xdim
        self.n = nbases
        self.W = self._weightsamples()

        if (lenscale_init.shape != (Xdim,)) and (lenscale_init.shape[0] != 1):
            raise ValueError("Parameter dimension doesn't agree with Xdim!")

        self.params = lenscale_init

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the random RBF to X.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: scalar or ndarray
            scalar or array of shape (d,) length scales (one for each dimension
            of X).

        Returns
        -------
        ndarray:
            of shape (N, 2*nbases) where nbases is number of random bases to
            use, given in the constructor.
        """

        N, D = X.shape
        lenscale = self._checkD(D, lenscale)

        WX = np.dot(X, safediv(self.W, lenscale))

        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(self.n)

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scales.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: scalar or ndarray
            scalar or array of shape (d,) length scales (one for each dimension
            of X).

        Returns
        -------
        ndarray:
            of shape (N, 2*nbases[, d]) where d is number of lenscales (if not
            ARD, i.e. scalar lenscale, this is just a 2D array). This is
            :math:`\partial \Phi(\mathbf{X}) / \partial \mathbf{l}`
        """

        N, D = X.shape
        lenscale = self._checkD(D, lenscale)

        WX = np.dot(X, safediv(self.W, lenscale))
        sinWX = - np.sin(WX)
        cosWX = np.cos(WX)

        dPhi = []
        for i, l in enumerate(lenscale):
            dWX = np.outer(X[:, i], - safediv(self.W[i, :], l**2))
            dPhi.append(np.hstack((dWX * sinWX, dWX * cosWX))
                        / np.sqrt(self.n))

        return np.dstack(dPhi) if len(lenscale) != 1 else dPhi[0]

    def _weightsamples(self):
        return np.random.randn(self.d, self.n)

    def _checkD(self, Xdim, lenscale):
        if Xdim != self.d:
            raise ValueError("Dimensions of data inconsistent!")

        # Promote dimension of lenscale
        if np.isscalar(lenscale):
            lenscale = np.array([lenscale])
        elif lenscale.ndim == 1:
            lenscale = lenscale[:, np.newaxis]

        if self.params.shape[0] != len(lenscale):
            raise ValueError("Dimensions of lenscale inconsistent!")

        return lenscale


class RandomLaplace(RandomRBF):
    """ 
    Random Laplace Basis -- Approximates a Laplace kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and 
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    """
    
    def _weightsamples(self):
        return cauchy.rvs(size=(self.d, self.n))


class RandomCauchy(RandomRBF):
    """ 
    Random Cauchy Basis -- Approximates a Cauchy kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Cauchy covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and 
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    """ 
    def _weightsamples(self):
        return laplace.rvs(size=(self.d, self.n))


class RandomMatern32(RandomRBF):
    """ 
    Random Matern 3/2 Basis -- Approximates a Matern 3/2 kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Matern covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and 
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    """ 

    def _weightsamples(self):
        # p = 1, v = 1.5 and the two is a transformation of variables between
        # Rasmussen 2006 p84 and the CF of a Student t (see wikipedia)
        return t.rvs(df=2 * 1.5, size=(self.d, self.n))


class RandomMatern52(RandomRBF):
    """ 
    Random Matern 5/2 Basis -- Approximates a Matern 5/2 kernel function

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Matern covariance function.

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and 
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    """ 

    def _weightsamples(self):
        # p = 2, v = 2.5 and the two is a transformation of variables between
        # Rasmussen 2006 p84 and the CF of a Student t (see wikipedia)
        return t.rvs(df=2 * 2.5, size=(self.d, self.n))


class FastFood(RandomRBF):
    """
    Fast Food basis function, which is an approximation of the random
    radial basis function for a large number of bases.

    This will make a linear regression model approximate a GP with an RBF
    covariance function.

    Parameters
    ----------
    nbases: int
        a scalar for how many random bases to create approximately, this
        actually will be to the neareset larger two power.
    Xdim: int   
        the dimension (d) of the observations.
    lenscale_init: Parameter, optional
        A scalar parameter to bound and initialise the length scales for
        optimization
    """

    @slice_init
    def __init__(self, nbases, Xdim, lenscale_init=Parameter(1., Positive())):

        self.params = lenscale_init

        # Make sure our dimensions are powers of 2
        l = int(np.ceil(np.log2(Xdim)))

        self.d = Xdim
        self.d2 = pow(2, l)
        self.k = int(np.ceil(nbases / self.d2))
        self.n = self.d2 * self.k

        # Draw consistent samples from the covariance matrix
        results = [self.__sample_matrices() for i in range(self.k)]
        self.B, self.G, self.PI, self.S = tuple(zip(*results))

    @slice_call
    def __call__(self, X, lenscale):
        """
        Apply the Fast Food RBF basis to X.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            of shape (N, 2*nbases) where nbases is number of random bases to
            use, given in the constructor (to nearest larger two power).
        """

        self._checkD(X.shape[1])

        VX = safediv(self.__makeVX(X), lenscale)
        Phi = np.hstack((np.cos(VX), np.sin(VX))) / np.sqrt(self.n)
        return Phi

    @slice_call
    def grad(self, X, lenscale):
        """
        Get the gradients of this basis w.r.t.\ the length scale.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        lenscale: float
            the length scale (scalar) of the RBFs to apply to X.

        Returns
        -------
        ndarray:
            shape (N, 2*nbases) where nbases is number of random RBF bases,
            again to the nearest larger two power. This is
            :math:`\partial \Phi(\mathbf{X}) / \partial l`
        """

        self._checkD(X.shape[1])

        VX = safediv(self.__makeVX(X), lenscale)
        dVX = - safediv(VX, lenscale)

        return np.hstack((-dVX * np.sin(VX), dVX * np.cos(VX))) \
            / np.sqrt(self.n)

    def __sample_matrices(self):

        B = np.random.randint(2, size=self.d2) * 2 - 1  # uniform from [-1,1]
        G = np.random.randn(self.d2)  # mean 0 std 1
        PI = np.random.permutation(self.d2)
        S = self._weightsamples(G)
        return B, G, PI, S

    def _weightsamples(self, G):
        return np.sqrt(gammaincinv(np.ceil(self.d2 / 2), 
                                       np.random.rand(self.d2)) / norm(G))
        # return np.sqrt(2 * gammaincinv(np.ceil(self.d2 / 2),
        #                                np.random.rand(self.d2))) / norm(G)


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

    def _checkD(self, D):
        if D != self.d:
            raise ValueError("Dimensions of data inconsistent!")


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
    def params(self):

        params = [b.params for b in self.bases if b.params != []]

        return params[0] if len(params) == 1 else params

    def __add__(self, other):

        if isinstance(other, BasisCat):
            return BasisCat(self.bases + other.bases)
        else:
            return BasisCat(self.bases + [other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)
