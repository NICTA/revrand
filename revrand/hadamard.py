"""
Fast Hadamard transform

Alistair Reid, NICTA 2015

The Hadamard transform is a recursive application of sums and differences
on a vector of length 2^n. The log(n) recursive application allow computation
in n log(n) operations instead of the naive n^2 per vector that would result
from computing and then multiplying a Hadamard matrix.

Note: the Walsh ordering is naturally computed by the recursive algorithm. The
sequence ordering requires a re-ordering afterwards incurring a similar
overhead to the original computation.

.. code:: python

    # Larger tests against calling Julia: [length 4*512 vectors:]
    M = np.argsort(np.sin(np.arange(64))).astype(float)
        Julia: 570us (sequence ordered)
        Al's single optimised: 560us (sequence ordered), 30us (natural)
        Al's basic vectorised: 1370us (sequence ordered)
"""


from __future__ import division

import numpy as np
from math import log


def hadamard(Y, ordering=True):
    """ *Very fast* Hadamard transform for single vector at a time

    Parameters
    ----------
    Y: ndarray
        the n*2^k data to be 1d hadamard transformed
    ordering: bool, optional
        reorder from Walsh to sequence

    Returns
    -------
    H: ndarray
        hadamard transformed data.
    """
    # dot is a sum product over the last axis of a and the second-to-last of b.
    # Transpose - can specify axes, default transposes a[0] and a[1] hmm
    n_vectors, n_Y = Y.shape
    matching = (n_vectors, 2, n_Y / 2)
    H = np.array([[1, 1], [1, -1]]) / 2.  # Julia uses 2 and not sqrt(2)?
    steps = int(log(n_Y) / log(2))
    assert(2**steps == n_Y)  # required
    for _ in range(steps):
        Y = np.transpose(Y.reshape(matching), (0, 2, 1)).dot(H)
    Y = Y.reshape((n_vectors, n_Y))
    if ordering:
        Y = Y[:, sequency(n_Y)]
    return Y


def sequency(length):
    # http://fourier.eng.hmc.edu/e161/lectures/wht/node3.html
    # Although this incorrectly explains grey codes...
    s = np.arange(length).astype(int)
    s = (s >> 1) ^ s  # Grey code ...
    # Reverse bits
    order = np.zeros(s.shape).astype(int)
    n = int(1)
    m = length // 2
    while n < length:
        order |= m * (n & s) // n
        n <<= 1
        m >>= 1
    return order


def hadamard_basic(Y, ordering=True):
    """
    Fast Hadamard transform using vectorised indices

    Parameters
    ----------
    Y: ndarray
        the n*2^k data to be 1d hadamard transformed
    ordering: bool, optional
        reorder from Walsh to sequence

    Returns
    -------
    H: ndarray
        hadamard transformed data.
    """

    Y = Y.T
    n = len(Y) / 2
    r = np.arange(len(Y))
    while n >= 1:
        ind = (r / n).astype(int) % 2 == 0
        a = (Y[ind] + Y[~ind]) / 2.
        b = (Y[ind] - Y[~ind]) / 2.
        Y[ind] = a
        Y[~ind] = b
        n /= 2
    if ordering:
        Y = Y[sequency(len(Y))]
    return Y.T
