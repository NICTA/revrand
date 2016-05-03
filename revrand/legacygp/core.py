""" Core functionality for GP library,
    Including covariance function composition, printing, introspection.
    Now jitchol free (I'm using SVD)
"""
import numpy as np
from . import linalg
from . import dtypes
# Now using the revrand optimize toolkit
from scipy.optimize import minimize
from revrand.optimize import structured_minimizer
from revrand.btypes import Bound, Parameter
import logging

log = logging.getLogger('GP')


def describe(kerneldef, hypers):
    return get_repr(kerneldef)(hypers)


def learn(X, y, kerneldef, opt_criterion=None, verbose=False, ftol=1e-8,
          maxiter=10000):

    n, d = X.shape

    if opt_criterion is None:
        opt_criterion = criterions.negative_log_marginal_likelihood
    else:
        pass  # check type

    cov_fn = compose(kerneldef)

    # Automatically determine the range
    meta = get_meta(kerneldef)

    params = [Parameter(i, Bound(l, h)) for i, l, h in zip(meta.initial_val,
                                                           meta.lower_bound,
                                                           meta.upper_bound)]

    def criterion(*theta):
        K = cov_fn(X, X, theta, True)  # learn with noise!
        factors = np.linalg.svd(K)
        value = opt_criterion(y, factors)
        if verbose:
            log.info("[{0}] {1}".format(value, theta))
        return value

    # up to here
    nmin = structured_minimizer(minimize)
    result = nmin(criterion, params, tol=ftol, options={'maxiter': maxiter},
                  jac=False, method='L-BFGS-B')
    print(result)
    return result.x


class criterions:

    # Compute the log marginal likelihood
    @staticmethod
    def negative_log_marginal_likelihood(y, factors):
        n = y.shape[0]
        nll = 0.5 * (linalg.svd_yKy(factors, y) +
                     linalg.svd_log_det(factors) +
                     n * np.log(2.0 * np.pi))
        return nll

    # Compute the log marginal likelihood
    @staticmethod
    def stacked_negative_log_marginal_likelihood(y, factors):
        n, n_stacks = y.shape
        nll = 0.5 * (np.asarray([linalg.svd_yKy(factors, y[:, i_stacks])
                     for i_stacks in range(n_stacks)]).sum() +
                     n_stacks * linalg.svd_log_det(factors) +
                     n_stacks * n * np.log(2.0 * np.pi))
        return nll

    # # Compute the leave one out neg log prob
    # @staticmethod
    # def negative_log_prob_cross_val(y, factors):
    #     n = y.shape[0]
    #     Kinv = svd_inverse(factors)  # more expensive!
    #     alpha = Kinv.dot(y)  # might as well now
    #     logprob = 0.
    #     for i in range(n):
    #         Kinvii = Kinv[i][i]
    #         mu_i = Y[i] - alpha[i]/Kinvii
    #         sig2i = 1/Kinvii
    #         logprob += stats.norm.logpdf(Y[i], loc=mu_i, scale=sig2i)
    #     return -logprob


def condition(X, y, kerneldef, hypers):
    """ Conditions a GP kernelFn(hypers) on the data X, y
        Arguments:
            Array nxd: X target input features
            Array nx1: y target outputs
            kernel
            hypers: vector of hyperparameters.
    """
    kernelFn = compose(kerneldef)
    kernel = lambda x1, x2, noise: kernelFn(x1, x2, hypers, noise)
    K = kernel(X, X, True)
    svd_factors = np.linalg.svd(K)
    alpha = linalg.svd_solve(svd_factors, y)
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


def mean(query_params):
    """ Computes the predictive mean for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nx1): the mean prediction.
    """
    assert(isinstance(query_params, dtypes.QueryParams))
    return np.dot(query_params.K_xxs.T, query_params.regressor.alpha)


def covariance(query_params, noise=True):
    """ Computes a full predictive covariance metrix for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nxn): the full covariance matrix.
    """
    assert(isinstance(query_params, dtypes.QueryParams))
    regressor = query_params.regressor
    K_xs = regressor.kernel(query_params.Xs, query_params.Xs, noise)
    v = linalg.svd_half_solve(regressor.factorisation, query_params.K_xxs)
    return K_xs - np.dot(v.T, v)


def variance(query_params, noise=True):
    """ Computes a full predictive covariance metrix for a query.
    Arguments:
        QueryParams: made with query()
    Returns:
        Array(nxn): the full covariance matrix.
    """
    assert(isinstance(query_params, dtypes.QueryParams))
    regressor = query_params.regressor
    K_xs = regressor.kernel(query_params.Xs, None, noise)
    v = linalg.svd_half_solve(regressor.factorisation, query_params.K_xxs)
    return K_xs - np.sum(v**2, axis=0)


def compose(kerneldef):
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
        return kerneldef(h, k)

    return user_kernel


def get_meta(kerneldef):
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

    kerneldef(h, k)  # Call the kernel def to pull out the hypers

    return dtypes.Range(mins, maxs, mids)


def get_repr(kerneldef):
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
            """ Returns a string
            """
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
            txta = PrintTree_.txt_(a)
            txtb = PrintTree_.txt_(b)

            if '+' in txta:
                txta = '(' + txta + ')'
            if '+' in txtb:
                txtb = '(' + txtb + ')'
            return PrintTree_(txta + '*' + txtb)


        def __radd__(b, a):
            return PrintTree_(PrintTree_.txt_(a) + '+' + PrintTree_.txt_(b))

    def print_fn(theta):
        theta_iter = iter(theta)
        h = lambda a, b, c=None: next(theta_iter)
        k = lambda f, par: PrintTree_(f.__name__ + '{' + PrintTree_.txt_(par)
                                      + '}')
        # now kerneldef(h, k) returns a print tree
        return kerneldef(h, k).__repr__()

    return print_fn  # We are returning a closure


def noise_kernel(func):
    func.__is_noise = True
    return func


def is_noise(func):
    return hasattr(func, '__is_noise')
