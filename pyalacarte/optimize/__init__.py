"""
The :mod:`pyalacarte.optimize` module provides methods for optimization.
"""

from .sgd import sgd
from .base import minimize

__all__ = ['sgd',
           'minimize']