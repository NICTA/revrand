""" Definitions of built-in kernels.

These Kernels are of the form:
  funcname(x_p, x_q, par)
Arguments:
    x_p (array n1*d)
    x_q (array n2*d)
Returns:
    if x_q is none:
        array(n1) : diagonal of K(x_p, x_p)
    otherwise:
    array(n1*n2): full covariance matrix
"""
import numpy as np
from scipy.spatial.distance import cdist
import logging
from .core import noise_kernel
log = logging.getLogger(__name__)


@noise_kernel
def lognoise(x_p, x_q, sigma):
    n = x_p.shape[0]
    if x_p is x_q:
        return np.exp(sigma) * np.eye(n)
    elif x_q is None:
        return np.exp(sigma) * np.ones(n)
    else:
        return 0.


def gaussian(x_p, x_q, LS):
    # The 'squared exponential' gaussian radial basis function kernel.
    # This kernel is known to be smooth, differentiable and stationary.
    if x_q is None:
        return np.ones(x_p.shape[0])
    deltasq = cdist(x_p/LS, x_q/LS, 'sqeuclidean')
    value = np.exp(-0.5 * deltasq)
    return value


def laplace(x_p, x_q, LS):
    if x_q is None:
        return np.ones(x_p.shape[0])
    deltasq = cdist(x_p / np.sqrt(LS), x_q / np.sqrt(LS), 'sqeuclidean')
    value = np.exp(- deltasq)
    return value


def sin(x_p, x_q, params):
    # The gaussian-enveloped sinusoidal kernel is good for modeling locally
    # oscillating series.
    if x_q is None:
        return np.ones(x_p.shape[0])
    freq, LS = params
    deltasq = cdist(x_p/LS, x_q/LS, 'sqeuclidean')
    value = np.exp(-0.5 * deltasq)*np.cos(np.sqrt(deltasq))
    return value


def matern3on2(x_p, x_q, LS):
    # The Matern 3/2 kernel is often used as a less smooth alternative to the
    # gaussian kernel for natural data.
    if x_q is None:
        return np.ones(x_p.shape[0])

    r = cdist(x_p/LS, x_q/LS, 'euclidean')
    value = (1.0 + r)*np.exp(-r)
    return value


def chisquare(x_p, x_q, eps=1e-5):

    if x_q is None:
        return np.ones(x_p.shape[0])

    x_pd = x_p[:, np.newaxis, :]
    x_qd = x_q[np.newaxis, :, :]

    return 2 * (x_pd * x_qd / (x_pd + x_qd + eps)).sum(axis=-1)
