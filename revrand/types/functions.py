"""Function utilities."""

from collections import namedtuple
from operator import neg


class FuncRes(namedtuple('FuncRes', ['value', 'grad'])):

    def __neg__(self):
        return self.__class__(value=neg(self.value), grad=tuple(map(neg, self.grad)))


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
        return -func(*args, **kwargs)

    return new_func
