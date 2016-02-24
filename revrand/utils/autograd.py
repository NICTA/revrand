"""
Some autograd convenience wrappers.

TODO: Should contribute these to autograd at some point.
"""

from autograd.core import getval
from autograd import multigrad


def multigrad_and_aux(fun, argnums=[0]):

    def multigrad_and_aux_fun(*args, **kwargs):
        saved = lambda: None
        def return_val_save_aux(*args, **kwargs):
            val, saved.aux = fun(*args, **kwargs)
            return val
        gradval = multigrad(return_val_save_aux, argnums)(*args, **kwargs)
        return gradval, saved.aux

    return multigrad_and_aux_fun


def value_and_multigrad(fun, argnums=[0]):

    def double_val_fun(*args, **kwargs):
        val = fun(*args, **kwargs)
        return val, getval(val)
    gradval_and_val = multigrad_and_aux(double_val_fun, argnums)
    flip = lambda x, y: (y, x)
    return lambda *args, **kwargs: flip(*gradval_and_val(*args, **kwargs))
