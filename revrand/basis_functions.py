"""
Various basis function objects specialised for parameter learning.

To make a new basis object, see the documentation of the Basis class.

"""

from __future__ import division

import sys
import inspect

import numpy as np
from six import wraps
from decorator import decorator  # Preserves function signature (pyth2 compat)
from scipy.linalg import norm
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from .btypes import Positive, Bound, Parameter
from .mathfun.linalg import hadamard
from .utils import issequence, atleast_list


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
    >>> base.transform(X).shape
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
def slice_transform(func, self, X, *vargs, **kwargs):
    """
    Decorator for implementing partial application.

    This must decorate the :code:`transform` and :code:`grad` methods of basis
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
    >>> Phi = base.transform(X, 1., np.ones(d))
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
    if issequence(grad):
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
    The base Basis class.

    To make other basis classes, make sure they are subclasses of this class to
    enable concatenation and operation with the machine learning algorithms.

    Example
    -------
    Basis concatentation works as follows if you subclass this class:

    >>> base = MyBasis1(properties1) + MyBasis2(properties2)  # doctest: +SKIP
    """

    _params = Parameter()

    @slice_init
    def __init__(self):
        """
        Construct this an instance of this class.

        This is also a good place to set non-learnable properties, and bounded
        Parameter types. An example Basis class with parameters may be

        Example:

        .. code-block:: python

            def __init__(self, property, param_init=Parameter(1, Bound())):

                self.property = property
                self.params = param_init

        All the :code:`params` property does is inform algorithms of the
        intitial value and any bounds this basis object has. This will need to
        correspond to any parameters input into the :code:`transform` and
        :code:`grad` methods. All basis class objects MUST have a
        :code:`params` property, which is either:
        - one Parameter object for an optimisable parameter, see btypes.py.
          Parameter objects with :code:`[]` values are interpreted as having no
          parameters.
        - a list of Parameter objects, one for each optimisable parameter
        """
        pass

    @slice_transform
    def transform(self, X):
        """
        Return the basis function applied to X.

        I.e. Phi(X, params), where params can also optionally be used and
        learned.

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

    @slice_transform
    def grad(self, X):
        """
        Return the gradient of the basis function for each parameter.

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

    @slice_transform
    def get_dim(self, X):
        """
        Get the output dimensionality of this basis.

        This makes a cheap call to transform with the initial parameter values
        to ascertain the dimensionality of the output features.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.

        Returns
        -------
        int:
            The dimensionality of the basis.
        """
        D = self.transform(X[[0]], *self.get_init_params()).shape[1]
        return D

    def get_init_params(self):
        return [p.value for p in atleast_list(self.params) if p.value != []]

    def _transform_popargs(self, X, *args):

        selfargs, otherargs = self._splitargs(args, self.transform, offset=1)

        return self.transform(X, *selfargs), otherargs

    def _grad_popargs(self, X, *args):

        selfargs, otherargs = self._splitargs(args, self.grad, offset=1)

        return self.grad(X, *selfargs), otherargs, selfargs

    def _splitargs(self, args, fn, offset=0):

        nargs = count_args(fn) - offset
        selfargs, otherargs = args[:nargs], args[nargs:]

        return selfargs, otherargs

    @property
    def params(self):
        """Get this object's Parameter types."""
        return self._params

    @params.setter
    def params(self, params):
        """Set this object's Parameter types."""
        self._params = params

    def __add__(self, other):

        return BasisCat([self, other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)


class BiasBasis(Basis):
    r"""
    Bias Basis for adding a bias term to a regressor.

    This just returns a column of a constant value so a bias term can be
    learned by a regressor.

    .. math::

        \phi(\mathbf{X}) = \mathbf{1} * \text{const}

    Parameters
    ----------
    offset: float, optional
        A scalar value to give the bias column. By default this is one.
    """

    @slice_init
    def __init__(self, offset=1.):

        self.offset = offset

    @slice_transform
    def transform(self, X):
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
            of shape (N, 1) of ones * self.offset.
        """
        N = len(X)
        return np.ones((N, 1)) * self.offset


class LinearBasis(Basis):
    r"""
    Linear basis class, basically this just prepends a columns of ones onto X.

    .. math::

        \phi(\mathbf{X}) = [\mathbf{1}, \mathbf{X}]

    Parameters
    ----------
    onescol: bool, optional
        If true, prepend a column of ones onto X.
    """

    @slice_init
    def __init__(self, onescol=False):

        self.onescol = onescol

    @slice_transform
    def transform(self, X):
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
    r"""
    Polynomial basis class.

    This essentially creates the concatenation,

    .. math::

        \phi(\mathbf{X}) = [\mathbf{1}, \mathbf{X}^1, \ldots, \mathbf{X}^p]

    where :math:`p` is the :code:`order` of the polynomial.

    Parameters
    ----------
    order: int
        the order of the polynomial to create.
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

    @slice_transform
    def transform(self, X):
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
    r"""
    Radial basis class.

    .. math::

        \phi(\mathbf{X}) =
            \exp \left( -\frac{\|\mathbf{X} - \mathbf{C}\|^2} {2 l^2} \right)

    Where :math:`\mathbf{C}` are radial basis centres, and :math:`l` is a
    length scale.

    Parameters
    ----------
    centres: ndarray
        array of shape (Dxd) where D is the number of centres for the
        radial bases, and d is the dimensionality of X.
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.

    Note
    ----
    This will have relevance vector machine-like behaviour with uncertainty.
    """

    @slice_init
    def __init__(self, centres, lenscale_init=Parameter(1., Positive())):

        self.M, self.d = centres.shape
        self.C = centres
        self._init_lenscale(lenscale_init)

    @slice_transform
    def transform(self, X, lenscale):
        """
        Apply the RBF to X.

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
            of shape (N, D) where D is number of RBF centres.
        """
        N, d = X.shape
        lenscale = self._checkdim(d, lenscale)

        den = (2 * lenscale**2)
        return np.exp(- cdist(X / den, self.C / den, 'sqeuclidean'))

    @slice_transform
    def grad(self, X, lenscale):
        r"""
        Get the gradients of this basis w.r.t.\ the length scale.

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
            of shape (N, D) where D is number of RBF centres. This is
            :math:`\partial \Phi(\mathbf{X}) / \partial l`
        """
        N, d = X.shape
        lenscale = self._checkdim(d, lenscale)

        Phi = self.transform(X, lenscale)
        dPhi = []
        for i, l in enumerate(lenscale):
            ldist = cdist(X[:, [i]] / l**3, self.C[:, [i]] / l**3,
                          'sqeuclidean')
            dPhi.append(Phi * ldist)

        return np.dstack(dPhi) if len(lenscale) != 1 else dPhi[0]

    def _init_lenscale(self, lenscale_init):

        if (lenscale_init.shape != (self.d,)) \
                and not np.isscalar(lenscale_init.value):
            raise ValueError("Parameter dimension doesn't agree with X"
                             " dimensions!")

        self.params = lenscale_init

    def _checkdim(self, Xdim, param, paramind=None):

        if Xdim != self.d:
            raise ValueError("Dimensions of data inconsistent!")

        # Promote dimension of parameter
        if np.isscalar(param):
            param = np.array([param])

        sparam = self.params if paramind is None else self.params[paramind]

        if (np.isscalar(sparam.value) and len(param) == 1) \
                or np.shape(param) == sparam.shape:
            return param
        else:
            raise ValueError("Dimensions of basis parameter is inconsistent!")


class SigmoidalBasis(RadialBasis):
    r"""
    Sigmoidal Basis.

    .. math::

        \phi(\mathbf{X}) =
            \sigma \left( \frac{\|\mathbf{X} - \mathbf{C}\|}{l} \right)

    where :math:`\mathbf{C}` are sigmoidal basis centres, :math:`l` is a
    length scale and :math:`\sigma` is the logistic sigmoid function defined by

    .. math::

        \sigma(a) = \frac{1}{1+e^{-a}}.

    Parameters
    ----------
    centres: ndarray
        array of shape (Dxd) where D is the number of centres for the bases,
        and d is the dimensionality of X.
    lenscale_init: Parameter, optional
        A scalar parameter to bound and initialise the length scales for
        optimization.
    """

    @slice_transform
    def transform(self, X, lenscale):
        r"""
        Apply the sigmoid basis function to X.

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
        N, d = X.shape
        lenscale = self._checkdim(d, lenscale)

        return expit(cdist(X / lenscale, self.C / lenscale, 'euclidean'))

    @slice_transform
    def grad(self, X, lenscale):
        r"""
        Get the gradients of this basis w.r.t.\  the length scale.

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
        N, d = X.shape
        lenscale = self._checkdim(d, lenscale)

        Phi = self.transform(X, lenscale)
        dPhi = []
        for i, l in enumerate(lenscale):
            ldist = cdist(X[:, [i]] / l**2, self.C[:, [i]] / l**2, 'euclidean')
            dPhi.append(- ldist * Phi * (1 - Phi))

        return np.dstack(dPhi) if len(lenscale) != 1 else dPhi[0]


class RandomRBF(RadialBasis):
    r"""
    Random RBF Basis -- Approximates an RBF kernel function.

    This will make a linear regression model approximate a GP with an
    (optionally ARD) RBF covariance function,

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \exp\left( -\frac{\| \mathbf{x} - \mathbf{x}' \|^2}{2 l^2} \right)

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    @slice_init
    def __init__(self,
                 nbases,
                 Xdim,
                 lenscale_init=Parameter(1., Positive()),
                 random_state=None
                 ):

        self.d = Xdim
        self.n = nbases
        self._random = check_random_state(random_state)
        self.W = self._weightsamples()
        self._init_lenscale(lenscale_init)

    @slice_transform
    def transform(self, X, lenscale):
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
        lenscale = self._checkdim(D, lenscale)[:, np.newaxis]

        WX = np.dot(X, self.W / lenscale)

        return np.hstack((np.cos(WX), np.sin(WX))) / np.sqrt(self.n)

    @slice_transform
    def grad(self, X, lenscale):
        r"""
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
        lenscale = self._checkdim(D, lenscale)[:, np.newaxis]

        WX = np.dot(X, self.W / lenscale)
        sinWX = - np.sin(WX)
        cosWX = np.cos(WX)

        dPhi = []
        for i, l in enumerate(lenscale):
            dWX = np.outer(X[:, i], - self.W[i, :] / l**2)
            dPhi.append(np.hstack((dWX * sinWX, dWX * cosWX)) /
                        np.sqrt(self.n))

        return np.dstack(dPhi) if len(lenscale) != 1 else dPhi[0]

    def _weightsamples(self):
        weights = self._random.randn(self.d, self.n)
        return weights


class RandomLaplace(RandomRBF):
    r"""
    Random Laplace Basis -- Approximates a Laplace kernel function.

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Laplace covariance function.

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \exp\left( -\frac{\| \mathbf{x} - \mathbf{x}' \|}{l} \right)

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    def _weightsamples(self):
        weights = self._random.standard_cauchy(size=(self.d, self.n))
        return weights


class RandomCauchy(RandomRBF):
    r"""
    Random Cauchy Basis -- Approximates a Cauchy kernel function.

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Cauchy covariance function.

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \frac{1}{1 + (\| \mathbf{x} - \mathbf{x}' \| / l)^2}

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    def _weightsamples(self):

        # A draw from a (regular) mv laplace is the same as:
        # X ~ Norm(mu, cov)
        # Z ~ gamma(1)
        # Y ~ (2 * Z) * X
        # See "Multivariate Generalized Laplace Distributions and Related
        # Random Fields":
        #   http://www.math.chalmers.se/Math/Research/Preprints/2010/47.pdf
        X = self._random.randn(self.d, self.n)
        Z = self._random.standard_gamma(1., size=(1, self.n))
        return X * np.sqrt(2 * Z)


class RandomMatern32(RandomRBF):
    r"""
    Random Matern 3/2 Basis -- Approximates a Matern 3/2 kernel function.

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Matern covariance function.

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \left(1 + \sqrt{3} \frac{\| \mathbf{x} - \mathbf{x}' \|}{l} \right)
            \exp
            \left(- \sqrt{3} \frac{\| \mathbf{x} - \mathbf{x}' \|}{l} \right)

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    def _weightsamples(self):
        return self._maternweight(p=1)

    def _maternweight(self, p):

        # p is the matern number (v = p + .5) and the two is a transformation
        # of variables between Rasmussen 2006 p84 and the CF of a Multivariate
        # Student t (see wikipedia). Also see "A Note on the Characteristic
        # Function of Multivariate t Distribution":
        #   http://ocean.kisti.re.kr/downfile/volume/kss/GCGHC8/2014/v21n1/
        #   GCGHC8_2014_v21n1_81.pdf
        # To sample from a m.v. t we use the formula
        # from wikipedia, x = y * np.sqrt(df / u) where y ~ norm(0, I),
        # u ~ chi2(df), then x ~ mvt(0, I, df)
        df = 2 * (p + 0.5)
        y = self._random.randn(self.d, self.n)
        u = self._random.chisquare(df, size=(self.n,))
        return y * np.sqrt(df / u)


class RandomMatern52(RandomMatern32):
    r"""
    Random Matern 5/2 Basis -- Approximates a Matern 5/2 kernel function.

    This will make a linear regression model approximate a GP with an
    (optionally ARD) Matern covariance function.

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \left(1 + \sqrt{5} \frac{\| \mathbf{x} - \mathbf{x}' \|}{l}
                + \frac{5 \| \mathbf{x} - \mathbf{x}' \|^2}{3l^2}
            \right)
            \exp
            \left(- \sqrt{5} \frac{\| \mathbf{x} - \mathbf{x}' \|}{l} \right)

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        how many unique random bases to create (twice this number will be
        actually created, i.e. real and imaginary components for each base)
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    def _weightsamples(self):
        return self._maternweight(p=2)


class FastFoodRBF(RandomRBF):
    r"""
    Fast Food radial basis function.

    This is an approximation of the random radial basis function for a large
    number of bases.

    .. math::

        \phi(\mathbf{x})^\top \phi(\mathbf{x}') \approx
            \exp\left( -\frac{\| \mathbf{x} - \mathbf{x}' \|^2}{2 l^2} \right)

    with a length scale, :math:`l` (a vector in :math:`\mathbb{R}^D` for ARD).

    Parameters
    ----------
    nbases: int
        a scalar for how many (unique) random bases to create approximately,
        this actually will be to the nearest larger two power.
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. If this is shape (d,),
        ARD length scales will be expected, otherwise an isotropic lenscale is
        learned.
    random_state: None, int or RandomState, optional
        random seed
    """

    @slice_init
    def __init__(self,
                 nbases,
                 Xdim,
                 lenscale_init=Parameter(1., Positive()),
                 random_state=None
                 ):

        self._random = check_random_state(random_state)
        self._init_dims(nbases, Xdim)
        self._init_lenscale(lenscale_init)
        self._init_matrices()

    @slice_transform
    def transform(self, X, lenscale):
        """
        Apply the Fast Food RBF basis to X.

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
            use, given in the constructor (to nearest larger two power).
        """
        lenscale = self._checkdim(X.shape[1], lenscale)

        VX = self._makeVX(X / lenscale)
        Phi = np.hstack((np.cos(VX), np.sin(VX))) / np.sqrt(self.n)
        return Phi

    @slice_transform
    def grad(self, X, lenscale):
        r"""
        Get the gradients of this basis w.r.t.\ the length scale.

        parameters
        ----------
        x: ndarray
            (n, d) array of observations where n is the number of samples, and
            d is the dimensionality of x.
        lenscale: scalar or ndarray
            scalar or array of shape (d,) length scales (one for each dimension
            of x).

        returns
        -------
        ndarray:
            shape (n, 2*nbases) where nbases is number of random rbf bases,
            again to the nearest larger two power. This is
            :math:`\partial \phi(\mathbf{x}) / \partial l`
        """
        d = X.shape[1]
        lenscale = self._checkdim(d, lenscale)

        VX = self._makeVX(X / lenscale)
        sinVX = - np.sin(VX)
        cosVX = np.cos(VX)

        dPhi = []
        for i, l in enumerate(lenscale):
            indlen = np.zeros(d)
            indlen[i] = 1. / l**2
            dVX = - self._makeVX(X * indlen)  # FIXME make this more efficient?
            dPhi.append(np.hstack((dVX * sinVX, dVX * cosVX)) /
                        np.sqrt(self.n))

        return np.dstack(dPhi) if len(lenscale) != 1 else dPhi[0]

    def _init_dims(self, nbases, Xdim):

        # Make sure our dimensions are powers of 2
        l = int(np.ceil(np.log2(Xdim)))

        self.d = Xdim
        self.d2 = pow(2, l)
        self.k = int(np.ceil(nbases / self.d2))
        self.n = self.d2 * self.k

    def _init_matrices(self):

        # Draw consistent samples from the covariance matrix
        shape = (self.k, self.d2)
        self.B = self._random.randint(2, size=shape) * 2 - 1  # uniform [-1,1]
        self.G = self._random.randn(*shape)  # mean 0 std 1
        self.PI = np.array([self._random.permutation(self.d2)
                            for _ in range(self.k)])
        self.S = self._weightsamples()

    def _weightsamples(self):
        s = np.sqrt(self._random.chisquare(self.d2, size=self.G.shape))
        return self.d2 * s / norm(self.G, axis=1)[:, np.newaxis]

    def _makeVX(self, X):
        N, d0 = X.shape

        # Pad the dimensions of X to nearest 2 power
        X_dash = np.zeros((N, self.d2))
        X_dash[:, 0:d0] = X

        VX = []
        for B, G, PI, S in zip(*(self.B, self.G, self.PI, self.S)):
            vX = hadamard(X_dash * B[np.newaxis, :], ordering=False)
            vX = vX[:, PI] * G[np.newaxis, :]
            vX = hadamard(vX, ordering=False) * S[np.newaxis, :] * \
                np.sqrt(self.d2)
            VX.append(vX)

        return np.hstack(VX)


class FastFoodGM(FastFoodRBF):
    """
    A mixture component from a Gaussian spectral mixture kernel approximation.

    This implements a GM basis component from "A la Carte - Learning Fast
    Kernels". This essentially learns the form of a kernel function, and so has
    no explicit kernel representation!

    Parameters
    ----------
    nbases: int
        a scalar for how many (unique) random bases to create approximately,
        this actually will be to the nearest larger two power.
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    mean_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the component frequency means for optimization. This will
        always initialise (d,) means if a scalr bound is given, it is applied
        to all means.
    lenscale_init: Parameter, optional
        A scalar or vector of shape (1,) or (d,) Parameter to bound and
        initialise the length scales for optimization. This will always
        initialise ARD length scales, if a scalr bound is given, it is applied
        to all length scales.
    random_state: None, int or RandomState, optional
        random seed
    """

    @slice_init
    def __init__(self,
                 nbases,
                 Xdim,
                 mean_init=Parameter(0., Bound()),
                 lenscale_init=Parameter(1., Positive()),
                 random_state=None
                 ):

        self._random = check_random_state(random_state)
        self._init_dims(nbases, Xdim)
        self.params = [self._init_param(mean_init),
                       self._init_param(lenscale_init)]
        self._init_matrices()

    @slice_transform
    def transform(self, X, mean, lenscale):
        """
        Apply the spectral mixture component basis to X.

        Parameters
        ----------
        X: ndarray
            (N, d) array of observations where N is the number of samples, and
            d is the dimensionality of X.
        mean: ndarray
            array of shape (d,) frequency means (one for each dimension of X).
        lenscale: ndarray
            array of shape (d,) length scales (one for each dimension of X).

        Returns
        -------
        ndarray:
            of shape (N, 4*nbases) where nbases is number of random bases to
            use, given in the constructor (to nearest larger two power).
        """
        mean = self._checkdim(X.shape[1], mean, paramind=0)
        lenscale = self._checkdim(X.shape[1], lenscale, paramind=1)

        VX = self._makeVX(X / lenscale)
        mX = X.dot(mean)[:, np.newaxis]
        Phi = np.hstack((np.cos(VX + mX), np.sin(VX + mX),
                         np.cos(VX - mX), np.sin(VX - mX))) / \
            np.sqrt(2 * self.n)

        return Phi

    @slice_transform
    def grad(self, X, mean, lenscale):
        r"""
        Get the gradients of this basis w.r.t.\ the mean and length scales.

        parameters
        ----------
        x: ndarray
            (n, d) array of observations where n is the number of samples, and
            d is the dimensionality of x.
        mean: ndarray
            array of shape (d,) frequency means (one for each dimension of X).
        lenscale: ndarray
            array of shape (d,) length scales (one for each dimension of X).

        returns
        -------
        ndarray:
            shape (n, 4*nbases) where nbases is number of random rbf bases,
            again to the nearest larger two power. This is
            :math:`\partial \phi(\mathbf{x}) / \partial mu`
        ndarray:
            shape (n, 4*nbases) where nbases is number of random rbf bases,
            again to the nearest larger two power. This is
            :math:`\partial \phi(\mathbf{x}) / \partial l`
        """
        d = X.shape[1]
        mean = self._checkdim(d, mean, paramind=0)
        lenscale = self._checkdim(d, lenscale, paramind=1)

        VX = self._makeVX(X / lenscale)
        mX = X.dot(mean)[:, np.newaxis]

        sinVXpmX = - np.sin(VX + mX)
        sinVXmmX = - np.sin(VX - mX)
        cosVXpmX = np.cos(VX + mX)
        cosVXmmX = np.cos(VX - mX)

        dPhi_len = []
        dPhi_mean = []
        for i, l in enumerate(lenscale):

            # Means
            dmX = X[:, [i]]
            dPhi_mean.append(np.hstack((dmX * sinVXpmX, dmX * cosVXpmX,
                                        -dmX * sinVXmmX, -dmX * cosVXmmX)) /
                             np.sqrt(2 * self.n))

            # Lenscales
            indlen = np.zeros(d)
            indlen[i] = 1. / l**2
            dVX = - self._makeVX(X * indlen)  # FIXME make this more efficient?
            dPhi_len.append(np.hstack((dVX * sinVXpmX, dVX * cosVXpmX,
                                       dVX * sinVXmmX, dVX * cosVXmmX)) /
                            np.sqrt(2 * self.n))

        dPhi_mean = np.dstack(dPhi_mean) if d != 1 else dPhi_mean[0]
        dPhi_len = np.dstack(dPhi_len) if d != 1 else dPhi_len[0]
        return dPhi_mean, dPhi_len

    def _init_param(self, param):

        if param.shape == (self.d,):
            return param
        elif param.shape in ((), (1,)):
            return (Parameter(np.ones(self.d) * param.value, param.bounds))
        else:
            raise ValueError("Parameter dimension doesn't agree with X"
                             " dimensions!")


#
# Helper Functions
#

def spectralmixture(Xdim,
                    apply_ind=None,
                    bases_per_component=50,
                    ncomponents=5,
                    means_init=None,
                    lenscales_init=None,
                    random_state=None
                    ):
    """
    Make a Gaussian spectral mixture basis.

    This is a helper function for easily creating a Gaussian spectral mixture
    basis from multiple FasFoodGM mixture components. This implements the full
    Gaussian spectral mixture from "A la Carte - Learning Fast Kernels".

    Parameters
    ----------
    Xdim: int
        the dimension (d) of the observations (or the dimension of the slices
        if using apply_ind).
    apply_ind: slice, optional
        a slice or index into which columns of X to apply this basis to.
    bases_per_component: int, optional
        a scalar for how many (unique) random bases to create approximately per
        mixture component. Approximately 4x this number of non-unique bases
        will be created per component, so 50 with 5 components is approx 1000
        bases.
    ncomponents: int, optional
        Number of FastFoodGM components to use in the mixture.
    means_init: list of Parameter, optional
        A list of :code:`Parameter`, :code:`len(means_init) == ncomponents`,
        to pass to each of the :code:`FastFoodGM`'s :code:`mean_init`
        constructor values.
    lenscale_init: list of Parameter, optional
        A list of :code:`Parameter`, :code:`len(lenscales_init) ==
        ncomponents`, to pass to each of the :code:`FastFoodGM`'s
        :code:`lenscale_init` constructor values.
    random_state: None, int or RandomState, optional
        random seed

    Returns
    -------
    GausSpecMix: BasisCat
        A concatenation of :code:`FastFoodGM` bases to make the full mixture.
    """
    random = check_random_state(random_state)

    if means_init is None:
        # Random values with random offset
        if Xdim > 1:
            means_init = [Parameter(random.randn(Xdim) + random.randn(1),
                                    Bound()) for _ in range(ncomponents)]
        else:
            means_init = [Parameter(random.randn(), Bound())
                          for _ in range(ncomponents)]
    elif len(means_init) != ncomponents:
        raise ValueError("Number of mean Parameters has to be equal to "
                         "ncomponents!")

    if lenscales_init is None:
        # Expected value of 1, with not too much deviation about that value
        size = None if Xdim == 1 else (Xdim,)
        lenscales_init = [
            Parameter(random.standard_gamma(3, scale=1. / 3, size=size),
                      Positive())
            for _ in range(ncomponents)
        ]
    elif len(lenscales_init) != ncomponents:
        raise ValueError("Number of length scale Parameters has to be equal "
                         "to ncomponents!")

    # Initialise all of the bases
    mixtures = [FastFoodGM(Xdim=Xdim, nbases=bases_per_component, mean_init=m,
                           lenscale_init=l, apply_ind=apply_ind,
                           random_state=random_state + i)
                for i, (m, l) in enumerate(zip(means_init, lenscales_init))]

    # Concatenate and return
    return BasisCat(mixtures)


#
# Other basis construction objects and functions
#

class BasisCat(object):
    """A class that implements concatenation of bases."""

    def __init__(self, basis_list):

        self.bases = basis_list

    def transform(self, X, *params):

        Phi = []
        args = params

        for base in self.bases:
            phi, args = base._transform_popargs(X, *args)
            Phi.append(phi)

        return np.hstack(Phi)

    def grad(self, X, *params):

        # Establish a few dimensions
        N = X.shape[0]
        D = self.transform(X[[0], :], *params).shape[1]

        # Get all gradients
        args = list(params)
        grads = []
        dims = [0]

        for i, base in enumerate(self.bases):

            # evaluate gradient and deal with multiple parameter gradients by
            # keeping track of the basis index
            g, args, sargs = base._grad_popargs(X, *args)
            if not issequence(g):
                grads.append((i, g))
            else:
                grads.extend([(i, gg) for gg in g])

            # Get the basis dimensionality for padding later
            baseD = base.transform(X[[0], :], *sargs).shape[1]
            dims.append(baseD)

        # Padding indices
        endinds = np.cumsum(dims)

        # Now generate structured gradients with appropriate zero padding
        for i, g in grads:

            if len(g) == 0:
                continue

            # Pad the gradient with respect to the total basis dimensionality
            dPhi_dim = (N, D) if g.ndim < 3 else (N, D, g.shape[2])
            dPhi = np.zeros(dPhi_dim)
            dPhi[:, endinds[i]:endinds[i + 1]] = g

            yield dPhi

    def get_dim(self, X):

        return np.sum((b.get_dim(X) for b in self.bases))

    def get_init_params(self):

        return [v for b in self.bases for v in b.get_init_params()]

    @property
    def params(self):

        paramlist = [b.params for b in self.bases if b.params.value != []]

        if len(paramlist) == 0:
            return Parameter()
        else:
            return paramlist if len(paramlist) > 1 else paramlist[0]

    def __add__(self, other):

        if isinstance(other, BasisCat):
            return BasisCat(self.bases + other.bases)
        else:
            return BasisCat(self.bases + [other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)
