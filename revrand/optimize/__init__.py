"""
The :mod:`revrand.optimize` module provides a standardized interface to
``scipy.optimize``, and also supports custom optimization methods.
"""

from .sgd import sgd, AdaDelta, AdaGrad, Momentum, Adam, SGDUpdater
from .decorators import (structured_minimizer,
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
