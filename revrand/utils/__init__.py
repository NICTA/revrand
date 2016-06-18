"""
Reusable utility functions
"""

from .decorators import (vectorize_args, unvectorize_args, vectorize_result)
from .base import (flatten, unflatten, couple, decouple, nwise, map_indices,
                   append_or_extend, atleast_list, issequence)
from .random import check_random_state

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
    'append_or_extend',
    'atleast_list',
    'issequence',
    'check_random_state'
]
