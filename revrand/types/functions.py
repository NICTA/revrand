"""Function utilities."""

from collections import namedtuple
from operator import neg


class FuncRes(namedtuple('FuncRes', ['value', 'grad'])):
    pass


def func_value(func):

    def new_func(*args, **kwargs):
        return func(*args, **kwargs).value

    return new_func


def func_grad(func, argnum=0):

    def new_func(*args, **kwargs):
        return func(*args, **kwargs).grad[argnum]

    return new_func


def func_negate(func):

    def new_func(*args, **kwargs):
        result = func(*args, **kwargs)
        return FuncRes(value=neg(result.value), grad=tuple(map(neg, result.grad)))

    return new_func
