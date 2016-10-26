"""
The :mod:`revrand.optimize` module provides a standardized interface to
popular optimization libraries and tools, such as NLopt and ``scipy.optimize``,
and also supports custom optimization methods.
"""

from .sgd import sgd, AdaDelta, AdaGrad, Momentum, Adam, SGDUpdater
from .base import (structured_minimizer,
                   structured_sgd,
                   logtrick_minimizer,
                   logtrick_sgd)

__all__ = [
    'sgd',
    'AdaDelta',
    'AdaGrad',
    'Momentum',
    'Adam',
    'SGDUpdater',
    'structured_minimizer',
    'structured_sgd',
    'logtrick_minimizer',
    'logtrick_sgd'
]
