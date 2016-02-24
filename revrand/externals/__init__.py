"""
External bundled dependencies.

Include lightweight functionality from external packages instead of having
strict dependencies on heavy-duty packages.
"""
from .sklearn import check_random_state

__all__ = [
    'check_random_state'
]
