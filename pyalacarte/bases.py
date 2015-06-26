""" Various basis function objects specialised for parameter learning.

    Authors:    Daniel Steinberg, Lachlan McCalman
    Date:       8 May 2015
    Institute:  NICTA

"""

import numpy as np
from scipy.linalg import norm
from scipy.special import gammaincinv
from scipy.spatial.distance import cdist
from pyalacarte.hadamard import hadamard

# TODO:
# - Make dealing with vector params less clunky, at the moment they have to all
#   be concatenated and passed in as an *args


# Wish list
# - ARD Fast Food
# - ARD Radial Basis
# - Gaussian Spectral Mixture
# - Sparse code basis


#
# Basis objects
#

# To make a new basis object, use the follwing template:
#
#   class NewBasis(LinearBasis):
#
#       def __init__(self, your_args, maybe_bounds):
#
#           # The following two lines are required, even if there are no params
#           self.bounds = [(pairs-of, bounds-here), (one-for, each-param)]
#           self.nbounds = len(self.get_bounds())
#
#       def get_basis(self, X, params):
#
#           # Make your basis
#           Phi = ...
#           return Phi
#
#       def get_grad(self, X, params)
#
#           # Optionally return gradients w.r.t. the params
#           dPhi1 = ...
#           dPhi2 = ...
#           ...
#
#           # The return always has to be a list, even if it has one item
#           return [dPhi1, dPhi2, ...]
#
# NOTE: If you make a new basis object, inherit LinearBasis to get automatic
#       concatenation with other bases working, bounds setting/checking, etc.
# NOTE: If you have no parameters, still make sure you set self.bounds = []
# NOTE: The bounds are upper-lower pairs, with None for no limit.


class LinearBasis(object):

    def __init__(self, onescol=False):

        self.onescol = onescol
        self.bounds = []                        # NOTE: Set default para bounds
        self.nbounds = len(self.get_bounds())   # NOTE: Need this for checks

    def get_basis(self, X):

        N, D = X.shape
        return np.hstack((np.ones((N, 1)), X)) if self.onescol else X

    def get_grad(self, X):

        # A bit inefficient, but it generalises well...
        return [np.zeros(self.get_basis(X).shape)]

    def get_bounds(self):
        return self.bounds

    def set_bounds(self, boundslist):

        if len(boundslist) != self.nbounds:
            raise ValueError("Require {} pairs of upper and lower bounds!"
                             .format(self.nbounds))

    def __add__(self, other):

        return BasisCat([self, other])

    def __radd__(self, other):

        import julia
        j = julia.Julia()
        self.hd = julia.core.JuliaModuleLoader(j).load_module("Hadamard")
        return self if other == 0 else self.__add__(other)


class PolynomialBasis(LinearBasis):

    def __init__(self, order):

        self.bounds = []
        self.nbounds = len(self.get_bounds())

        if order < 0:
            raise ValueError("Polynomial order must be positive")
        self.order = order

    def get_basis(self, X):

        N, D = X.shape
        powarr = np.tile(np.arange(self.order+1), (D, 1))
        return (X[:, :, np.newaxis] ** powarr).reshape((N, D*(self.order+1)))


class FastFood(LinearBasis):
    """
        TODO:
            - Check dimensionality of X for consistency in get_basis and
              get_grad()
    """

    def __init__(self, nbases, Xdim, lenscaleLB=1e-7):

        self.bounds = [(lenscaleLB, None)]
        self.nbounds = len(self.get_bounds())

        # Make sure our dimensions are powers of 2
        l = int(np.ceil(np.log2(Xdim)))
        self.d = pow(2, l)
        self.k = int(np.ceil(nbases/self.d))
        self.n = self.d * self.k

        # Draw consistent samples from the covariance matrix
        results = [self.__sample_params() for i in range(self.k)]
        self.B, self.G, self.PI, self.S = tuple(zip(*results))

    def get_basis(self, X, lenscale):
        """X is a npoints x ndims matrix"""

        VX = self.__makeVX(X) / lenscale
        Phi = np.hstack((np.cos(VX), np.sin(VX))) / np.sqrt(self.n)
        return Phi

    def get_grad(self, X, lenscale):

        VX = self.__makeVX(X)
        dVX = - VX / lenscale**2
        VX /= lenscale

        return [np.hstack((-dVX * np.sin(VX), dVX * np.cos(VX)))
                / np.sqrt(self.n)]

    def __sample_params(self):

        B = np.random.randint(2, size=self.d) * 2 - 1  # uniform from [-1,1]
        G = np.random.randn(self.d)  # mean 0 std 1
        PI = np.random.permutation(self.d)
        S = np.sqrt(2 * gammaincinv(np.ceil(self.d/2),
                                    np.random.rand(self.d))) / norm(G)
        return B, G, PI, S

    def __makeVX(self, X):
        m, d0 = X.shape

        # Pad the dimensions of X to nearest 2 power
        X_dash = np.zeros((m, self.d))
        X_dash[:, 0:d0] = X

        VX = []
        for B, G, PI, S in zip(*(self.B, self.G, self.PI, self.S)):
            vX = hadamard(X_dash * B[np.newaxis, :], ordering=False)
            vX = vX[:, PI] * G[np.newaxis, :]
            VX.append(hadamard(vX, ordering=False) * S[np.newaxis, :]
                      * np.sqrt(self.d))

        return np.hstack(VX)


class RandomRBF(LinearBasis):
    """ Random RBF Basis, otherwise known as Random Kitchen Sinks."""

    def __init__(self, nbases, Xdim, lenscaleLB=1e-7):
        self.d = Xdim
        self.n = nbases
        self.W = np.random.randn(self.d, self.n)
        self.bounds = [(lenscaleLB, None)]
        self.nbounds = len(self.get_bounds())

    def get_basis(self, X, lenscale):

        N, D = X.shape
        self.__checkD(D)

        sig = 4 / (np.pi * lenscale)
        WX = np.dot(X, self.W * sig)

        return np.sqrt(2/self.n) * np.hstack((np.cos(WX), np.sin(WX)))

    def get_grad(self, X, lenscale):

        N, D = X.shape
        self.__checkD(D)

        sig = 4 / (np.pi * lenscale)
        dsig = - 4 / (np.pi * lenscale**2)
        WX = np.dot(X, self.W)
        dWX = WX * dsig
        WX *= sig

        return [np.hstack((-dWX * np.sin(WX), dWX * np.cos(WX)))
                * np.sqrt(2/self.n)]

    def __checkD(self, D):
        if D != self.d:
            raise ValueError("Dimensions of data inconsistent!")


class RandomRBF_ARD(LinearBasis):

    def __init__(self, nbases, Xdim, lenscaleLB=1e-7):

        self.d = Xdim
        self.n = nbases
        self.W = np.random.randn(self.d, self.n)
        self.bounds = [(lenscaleLB, None)] * self.d
        self.nbounds = len(self.get_bounds())

    def get_basis(self, X, *lenscales):

        N, D = X.shape
        self.__checkD(D, len(lenscales))

        sig = 4 / (np.pi * np.asarray(lenscales))[:, np.newaxis]
        WX = np.dot(X, sig * self.W)

        return np.sqrt(2/self.n) * np.hstack((np.cos(WX), np.sin(WX)))

    def get_grad(self, X, *lenscales):

        N, D = X.shape
        self.__checkD(D, len(lenscales))

        sig = 4 / (np.pi * np.asarray(lenscales))[:, np.newaxis]
        WX = np.dot(X, sig * self.W)
        sinWX = - np.sin(WX)
        cosWX = np.cos(WX)

        dPhi = []
        for i, l in enumerate(lenscales):
            dWX = np.outer(X[:, i], - 4 / (np.pi * l**2) * self.W[i, :])
            dPhi.append(np.hstack((dWX*sinWX, dWX*cosWX)) * np.sqrt(2/self.n))

        return dPhi

    def __checkD(self, Xdim, lendim):
        if Xdim != self.d:
            raise ValueError("Dimensions of data inconsistent!")
        if lendim != self.d:
            raise ValueError("Dimensions of lenscale inconsistent!")


class RadialBasis(LinearBasis):
    """ TODO

        NOTE: This will have relevance vector machine-like behaviour with
        uncertainty and for deaggregation tasks!
    """

    def __init__(self, centres, lenscaleLB=1e-7):

        self.M, self.D = centres.shape
        self.C = centres
        self.bounds = [(lenscaleLB, None)]
        self.nbounds = len(self.get_bounds())

    def get_basis(self, X, lenscale):

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        Phi = np.exp(- cdist(X, self.C, 'sqeuclidean') / (2 * lenscale**2))

        return Phi

    def get_grad(self, X, lenscale):

        N, D = X.shape
        if self.D != D:
            raise ValueError("X has inconsistent dimensionality!")

        sdist = cdist(X, self.C, 'sqeuclidean')
        dPhi = np.exp(- sdist / (2 * lenscale**2)) * sdist / lenscale**3

        return [dPhi]


#
# Other basis construction objects and functions
#

class BasisCat(object):
    """ A class that implements concatenation of bases. """

    def __init__(self, basis_list):

        self.bases = basis_list
        self.nhypers_list = [len(b.get_bounds()) for b in self.bases]
        self.nhypers = np.sum(self.nhypers_list)

    def get_basis(self, X, *hypers):

        phi = [b.get_basis(X, *h) for b, h in zip(self.bases,
                                                  self.__get_hypers(hypers))]
        return np.hstack(phi)

    def get_grad(self, X, *hypers):

        # Get the gradients from each basis in list of lists
        hlist = self.__get_hypers(hypers)
        grads = [Phi.get_grad(X, *h) for Phi, h in zip(self.bases, hlist)]

        # Now combine the padded arrays and gradient in correct positions
        dPhis = []
        for i, glist in enumerate(grads):

            # Ignore bases with no gradients
            if not hlist[i]:
                continue

            # Pad gradient with relevant zeros for other bases
            for g in glist:
                dPhi = [np.zeros(grads[j][0].shape) if j != i else g
                        for j in range(len(self.bases))]
                dPhis.append(np.hstack(dPhi))

        return dPhis

    def get_bounds(self):

        bounds = []
        for b in self.bases:
            bounds += b.get_bounds()

        return bounds

    def __add__(self, other):

        if isinstance(other, BasisCat):
            return BasisCat(self.bases + other.bases)
        else:
            return BasisCat(self.bases + [other])

    def __radd__(self, other):

        return self if other == 0 else self.__add__(other)

    def __get_hypers(self, hypers):

        if len(hypers) != self.nhypers:
            raise ValueError("Incorrect number of hyperparameters!")

        # Make sure we are dealing with lists
        hyp = list(hypers)

        # Get the number of hyperparameters per object in reverse
        nhypers_r = list(self.nhypers_list)
        nhypers_r.reverse()

        # Now chop up the hyperparameters list to feed into the relevant bases
        hypslist = [[hyp.pop() for n in range(nhyps)] for nhyps in nhypers_r]
        hypslist.reverse()
        return hypslist
