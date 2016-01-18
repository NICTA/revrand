"""
Reusable utility functions
"""

from .decorators import (vectorize_args, unvectorize_args, vectorize_result)
from .base import (flatten, unflatten, couple, decouple, nwise, map_indices)

__all__ = [
    'vectorize_args',
    'unvectorize_args',
    'vectorize_result',
    'flatten',
    'unflatten',
    'couple',
    'decouple',
    'nwise',
    'map_indices'
]
