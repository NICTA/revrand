"""
Reusable utility functions
"""

from .decorators import (vectorize_args, unvectorize_args, vectorize_result)
from .base import (flatten, unflatten, couple, decouple, nwise, map_indices,
                   atleast_list, issequence)
from .rand import endless_permutations

__all__ = [
    'vectorize_args',
    'unvectorize_args',
    'vectorize_result',
    'flatten',
    'unflatten',
    'couple',
    'decouple',
    'nwise',
    'map_indices',
    'atleast_list',
    'issequence',
    'endless_permutations'
]
