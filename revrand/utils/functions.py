"""Function utilities."""

from collections import namedtuple


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
