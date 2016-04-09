"""
The :mod:`revrand.basis_function` module implements basis functions
and mechanisms for creating higher-order basis functions.
"""

from .base import (slice_init,
                   slice_call,
                   apply_grad,
                   Basis,
                   LinearBasis,
                   PolynomialBasis,
                   RadialBasis,
                   SigmoidalBasis,
                   RandomRBF,
                   RandomRBF_ARD,
                   FastFood
                   )

__all__ = ['slice_init',
           'slice_call',
           'apply_grad',
           'Basis',
           'LinearBasis',
           'PolynomialBasis',
           'RadialBasis',
           'SigmoidalBasis',
           'RandomRBF',
           'RandomRBF_ARD',
           'FastFood',
           ]
