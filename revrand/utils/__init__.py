"""
Reusable utility functions
"""

from .math import softplus, softmax, logsumexp, safelog, safesoftplus, safediv
from .decorators import (vectorize_args, unvectorize_args, vectorize_result)
from .base import (flatten, unflatten, couple, decouple, nwise, map_indices,
                   append_or_extend, atleast_list)

__all__ = [
    'softplus',
    'softmax',
    'logsumexp',
    'safelog',
    'safesoftplus',
    'safediv',
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
    'atleast_list'
]
