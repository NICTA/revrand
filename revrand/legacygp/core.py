""" Core functionality for GP library,
    Including covariance function composition, printing, introspection.
    Now jitchol free (I'm using SVD)
"""
import numpy as np
from . import linalg
from . import dtypes
# Now using the revrand optimize toolkit
from revrand.optimize import minimize, structured_minimizer, Bound
# import scipy.linalg as la
import logging

log = logging.getLogger('GP')


def describe(kernel_defn, hypers):
    return get_repr(kernel_defn)(hypers)


def learn(X, y, kernel_defn, optCriterion=None, verbose=False, ftol=1e-8,
          maxiter=10000):

    n, d = X.shape

    if optCriterion is None:
        optCriterion = criterions.negative_log_marginal_likelihood
    else:
        pass  # check type

    cov_fn = compose(kernel_defn)

    # Automatically determine the range
    meta = get_meta(kernel_defn)
    bounds = [Bound(l, h) for l, h in zip(meta.lowerBound, meta.upperBound)]
    theta0 = meta.initialVal

    def criterion(*theta):
        K = cov_fn(X, X, theta, True)  # learn with noise!
        factors = np.linalg.svd(K)
        value = optCriterion(y, factors)
        if verbose:
            log.info("[{0}] {1}".format(value, theta))
        return value

    # up to here
    nmin = structured_minimizer(minimize)
    result = nmin(criterion, theta0, backend='scipy',
                  ftol=ftol, maxiter=maxiter, jac=False,
                  bounds=bounds, method='L-BFGS-B')
    print(result)
    return result.x


class criterions:

    # Compute the log marginal likelihood
    @staticmethod
    def negative_log_marginal_likelihood(y, factors):
        n = y.shape[0]
        nll = 0.5 * (linalg.svd_yKy(factors, y) +
                     linalg.svdLogDet(factors) +
                     n * np.log(2.0 * np.pi))
        return nll

    # Compute the log marginal likelihood
    @staticmethod
    def stacked_negative_log_marginal_likelihood(y, factors):
        n, n_stacks = y.shape
        nll = 0.5 * (np.asarray([linalg.svd_yKy(factors, y[:, i_stacks])
                     for i_stacks in range(n_stacks)]).sum() +
                     n_stacks * linalg.svdLogDet(factors) +
                     n_stacks * n * np.log(2.0 * np.pi))
        return nll

    # # Compute the leave one out neg log prob
    # @staticmethod
    # def negative_log_prob_cross_val(y, factors):
    #     n = y.shape[0]
    #     Kinv = svdInverse(factors)  # more expensive!
    #     alpha = Kinv.dot(y)  # might as well now
    #     logprob = 0.
    #     for i in range(n):
    #         Kinvii = Kinv[i][i]
    #         mu_i = Y[i] - alpha[i]/Kinvii
    #         sig2i = 1/Kinvii
    #         logprob += stats.norm.logpdf(Y[i], loc=mu_i, scale=sig2i)
    #     return -logprob


def condition(X, y, kernel_defn, hypers):
    """ Conditions a GP kernelFn(hypers) on the data X, y
        Arguments:
            Array nxd: X target input features
            Array nx1: y target outputs
            kernel
            hypers: vector of hyperparameters.
    """
    kernelFn = compose(kernel_defn)
    kernel = lambda x1, x2, noise: kernelFn(x1, x2, hypers, noise)
    K = kernel(X, X, True)
    svd_factors = np.linalg.svd(K)
    alpha = linalg.svdSolve(svd_factors, y)
    return dtypes.RegressionParams(X, svd_factors, alpha, kernel, y)


def query(regressor, Xs):
    """ Prepares a query object given a regressor and query points
    Arguments:
        RegressionParams: Regressor
        Array: Xs
    Returns:
        Query parameters
    """
    assert(isinstance(regressor, dtypes.RegressionParams))

    K_xxs = regressor.kernel(regressor.X, Xs, False)
    return dtypes.QueryParams(regressor, Xs, K_xxs)


def mean(queryParams):
    """ Computes the predictive mean for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nx1): the mean prediction.
    """
    assert(isinstance(queryParams, dtypes.QueryParams))
    return np.dot(queryParams.K_xxs.T, queryParams.regressor.alpha)


def covariance(queryParams, noise=True):
    """ Computes a full predictive covariance metrix for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nxn): the full covariance matrix.
    """
    assert(isinstance(queryParams, dtypes.QueryParams))
    regressor = queryParams.regressor
    K_xs = regressor.kernel(queryParams.Xs, queryParams.Xs, noise)
    v = linalg.svdHalfSolve(regressor.factorisation, queryParams.K_xxs)
    return K_xs - np.dot(v.T, v)


def variance(queryParams, noise=True):
    """ Computes a full predictive covariance metrix for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nxn): the full covariance matrix.
    """
    assert(isinstance(queryParams, dtypes.QueryParams))
    regressor = queryParams.regressor
    K_xs = regressor.kernel(queryParams.Xs, None, noise)
    v = linalg.svdHalfSolve(regressor.factorisation, queryParams.K_xxs)
    return K_xs - np.sum(v**2, axis=0)


def compose(kernel_defn):
    """ Converts a user's kernel definition fn(h,k)
        Args:
            h - Hyperparameter function (min, max, mid)
            k - Kernel call function (name, hyper, optional_list_of_dimensions)
        Returns:
            function k(xp,xq, theta)
    """
    def user_kernel(x1, x2, thetas, noise):
        theta_iter = iter(thetas)
        h = lambda a, b, c=None: next(theta_iter)
        k = lambda kfunc, par: (kfunc(x1, x2, par) if
                                (noise or not hasattr(kfunc, '__is_noise'))
                                else 0.)
        return kernel_defn(h, k)

    return user_kernel


def get_meta(kernel_defn):
    """ Introspects a user's kernel definition fn(h,k)
           h - Hyperparameter function (min, max, mid)
           k - Kernel call function (name, hyper, optional_list_of_dimensions)
        Returns:
           Range(min, max, intial) of lists.
    """
    mins = []
    mids = []
    maxs = []

    def h(min, max, initial=None):
        # Logs the inputs in a list and do nothing else.
        if initial is None:
            initial = 0.5*(min+max)
        mins.append(min)
        mids.append(initial)
        maxs.append(max)
        return 0.

    def k(fn, par):
        # Compatible 'kernel' that takes nothing and does nothing.
        return 0.

    kernel_defn(h, k)  # Call the kernel def to pull out the hypers

    return dtypes.Range(mins, maxs, mids)


def get_repr(kernel_defn):
    """ Returns a Function that prints a kernel to console with its
        hyperparameters.
    """

    class PrintTree_:
        # Object for recording the covariance function call tree.
        # (keep in private scope)
        def __init__(self, txt):
            self.txt = txt

        def __repr__(self):
            return self.txt

        @staticmethod
        def txt_(a):
            if isinstance(a, float):
                return "{0:.3f}".format(a)
            else:
                return a.__repr__()

        def __mul__(a, b):
            txta = PrintTree_.txt_(a)
            txtb = PrintTree_.txt_(b)

            if '+' in txta:
                txta = '(' + txta + ')'
            if '+' in txtb:
                txtb = '(' + txtb + ')'
            return PrintTree_(txta + '*' + txtb)

        def __add__(a, b):
            return PrintTree_(PrintTree_.txt_(a) + '+' + PrintTree_.txt_(b))

        def __rmul__(b, a):
            return PrintTree_.__mul__(a, b)

        def __radd__(b, a):
            return PrintTree_.__add__(a, b)

    def print_fn(theta):
        theta_iter = iter(theta)
        h = lambda a, b, c=None: next(theta_iter)
        k = lambda f, par: PrintTree_(f.__name__ + '{' + PrintTree_.txt_(par)
                                      + '}')
        # now kernel_defn(h, k) returns a print tree
        return kernel_defn(h, k).__repr__()

    return print_fn  # We are returning a closure


def noise_kernel(func):
    func.__is_noise = True
    return func


def isNoise(func):
    return hasattr(func, '__is_noise')



