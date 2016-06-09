import inspect
import numpy as np

from six.moves import map, range, zip
from functools import partial
from itertools import tee
from inspect import isgenerator


class Bunch(dict):
    """
    Container object for datasets: dictionary-like object that exposes
    its keys as attributes.

    Examples
    --------
    >>> b = Bunch(foo=42, bar=10)
    >>> b == {'foo': 42, 'bar': 10}
    True
    >>> b.foo
    42
    >>> b.bar
    10
    >>> b['foo']
    42
    >>> b.baz = 61
    >>> b.baz
    61
    >>> b['baz']
    61
    """
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def append_or_extend(mylist, *objs):
    """
    Append an object to a list, or extend the list if the object is a list.

    Parameters
    ----------
    mylist: list
        list to extend
    objs: object(s)
        object(s) to append to the list if not a list, else concatenate the
        lists

    Returns
    -------
    list
        the extended list.
    """

    for obj in objs:
        mylist.extend(obj if isinstance(obj, list) else [obj])
    return mylist


def issequence(obj):
    """
    Tests if an object is an iterable sequence of type generator, list or
    tuple.

    Parameters
    ----------
    obj: object
        object to test

    Returns
    -------
    bool:
        True if :code:`obj` is a tuple, list or generator

    Examples
    --------
    >>> issequence([1, 2])
    True
    >>> issequence((1,))
    True
    >>> issequence((i for i in range(8)))
    True
    >>> issequence(np.array([1, 2, 3]))
    False
    """

    return inspect.isgenerator(obj) or isinstance(obj, (list, tuple))


def atleast_list(a):
    """
    Promote an object to a list if not a list or generator

    Parameters
    ----------
    a: object
        any object you want to at least be a list with one element

    Returns
    -------
    list or generator:
        untounched if :code:`a` was a generator or list, otherwise :code:`[a]`.
    """

    return a if isinstance(a, list) or isgenerator(a) else [a]


def couple(f, g):
    """
    Given a pair of functions that take the same arguments, return a
    single function that returns a pair consisting of the return values
    of each function.

    Notes
    -----
    Equivalent to::

        lambda f, g: lambda *args, **kwargs: (f(*args, **kwargs),
                                              g(*args, **kwargs))

    Examples
    --------
    >>> f = lambda x: 2*x**3
    >>> df = lambda x: 6*x**2
    >>> f_new = couple(f, df)
    >>> f_new(5)
    (250, 150)

    """
    def coupled(*args, **kwargs):
        return f(*args, **kwargs), g(*args, **kwargs)
    return coupled


def decouple(fn):
    """
    Create two functions of one argument and one return from a function that
    takes two arguments and has two returns

    Examples
    --------
    >>> h = lambda x: (2*x**3, 6*x**2)
    >>> f, g = decouple(h)

    >>> f(5)
    250

    >>> g(5)
    150
    """
    def fst(*args, **kwargs):
        return fn(*args, **kwargs)[0]

    def snd(*args, **kwargs):
        return fn(*args, **kwargs)[1]

    return fst, snd


def nwise(iterable, n):
    """
    Iterator that acts like a sliding window of size `n`; slides over
    some iterable `n` items at a time. If iterable has `m` elements,
    this function will return an iterator over `m-n+1` tuples.

    Parameters
    ----------
    iterable : iterable
        An iterable object.

    n : int
        Window size.

    Returns
    -------
    iterator of tuples.
        Iterator of size `n` tuples

    Notes
    -----
    First `n` iterators are created::

        iters = tee(iterable, n)

    Next, iterator `i` is advanced `i` times::

        for i, it in enumerate(iters):
            for _ in range(i):
                next(it, None)

    Finally, the iterators are zipped back up again::

        return zip(*iters)

    Examples
    --------
    >>> a = [2, 5, 7, 4, 2, 8, 6]

    >>> list(nwise(a, n=3))
    [(2, 5, 7), (5, 7, 4), (7, 4, 2), (4, 2, 8), (2, 8, 6)]

    >>> pairwise = partial(nwise, n=2)
    >>> list(pairwise(a))
    [(2, 5), (5, 7), (7, 4), (4, 2), (2, 8), (8, 6)]

    >>> list(nwise(a, n=1))
    [(2,), (5,), (7,), (4,), (2,), (8,), (6,)]

    >>> list(nwise(a, n=7))
    [(2, 5, 7, 4, 2, 8, 6)]

    .. todo::

       These should probably raise `ValueError`...

    >>> list(nwise(a, 8))
    []

    >>> list(nwise(a, 9))
    []

    A sliding window of size `n` over a list of `m` elements
    gives `m-n+1` windows

    >>> len(a) - len(list(nwise(a, 2))) == 1
    True

    >>> len(a) - len(list(nwise(a, 3))) == 2
    True

    >>> len(a) - len(list(nwise(a, 7))) == 6
    True
    """

    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        for _ in range(i):
            next(it, None)
    return zip(*iters)

pairwise = partial(nwise, n=2)


def flatten(arys, order='C', returns_shapes=True):
    """
    Flatten a list of ndarrays and/or numpy scalars (0d-array) with possibly
    heterogenous dimensions and shapes and concatenate them together into a
    flat (1d) array.

    .. note::

       Not to be confused with `np.ndarray.flatten()` (a more befitting
       might be `chain` or `stack` or maybe something else entirely
       since this function is more than either `concatenate` or
       `np.flatten` itself. Rather, it is the composition of the former
       with the latter.

    Parameters
    ----------
    arys : list of array_like
        One or more input arrays of possibly heterogenous shapes and
        sizes.

    order : {'C', 'F', 'A'}, optional
        Whether to flatten in C (row-major), Fortran (column-major)
        order, or preserve the C/Fortran ordering. The default is 'C'.

    returns_shapes : bool, optional
        Default is `True`. If `True`, the tuple `(flattened, shapes)` is
        returned, otherwise only `flattened` is returned.

    Returns
    -------

    flattened,[shapes] : {1darray, list of tuples}
        Return the flat (1d) array resulting from the concatenation of
        flattened ndarrays. When `returns_shapes` is `True`, return a
        list of tuples containing also the shapes of each array as the
        second element.

    See Also
    --------
    revrand.utils.unflatten : its inverse

    Notes
    -----
    Equivalent to::

        lambda arys, order='C', returns_shapes=True: \
            (np.hstack(map(partial(np.ravel, order=order), ndarrays)),
             list(map(np.shape, ndarrays))) if returns_shapes \
             else np.hstack(map(partial(np.ravel, order=order), ndarrays))

    This implementation relies on the fact that scalars are treated as
    0-dimensional arrays. That is,

    >>> a = 4.6
    >>> np.ndim(a)
    0
    >>> np.shape(a)
    ()

    >>> np.ravel(a)
    array([ 4.6])

    Note also that the following is also a 0-dimensional array

    >>> b = np.array(3.14)
    >>> np.ndim(b)
    0
    >>> np.shape(b)
    ()

    .. important::

       When 0-dimensional arrays of the latter form are flattened,
       *they  will be unflattened as a scalar*. (Because special cases
       aren't special enough to break the rules.)

    Examples
    --------
    >>> a = 9
    >>> b = np.array([4, 7, 4, 5, 2])
    >>> c = np.array([[7, 3, 1],
    ...               [2, 6, 6]])
    >>> d = np.array([[[6, 5, 5],
    ...                [1, 6, 9]],
    ...               [[3, 9, 1],
    ...                [9, 4, 1]]])

    >>> flatten([a, b, c, d]) # doctest: +NORMALIZE_WHITESPACE
    (array([9, 4, 7, 4, 5, 2, 7, 3, 1, 2, 6, 6, 6, 5, 5, 1, 6, 9, 3, 9,
            1, 9, 4, 1]), [(), (5,), (2, 3), (2, 2, 3)])

    >>> flatten([a, b, c, d], order='F')
    ... # doctest: +NORMALIZE_WHITESPACE
    (array([9, 4, 7, 4, 5, 2, 7, 2, 3, 6, 1, 6, 6, 3, 1, 9, 5, 9, 6, 4,
            5, 1, 9, 1]), [(), (5,), (2, 3), (2, 2, 3)])

    Note that scalars and 0-dimensional arrays are treated differently
    from 1-dimensional singleton arrays.

    >>> flatten([3.14, np.array(2.71), np.array([1.61])])
    ... # doctest: +NORMALIZE_WHITESPACE
    (array([ 3.14,  2.71,  1.61]), [(), (), (1,)])

    >>> flatten([a, b, c, d], returns_shapes=False)
    ... # doctest: +NORMALIZE_WHITESPACE
    array([9, 4, 7, 4, 5, 2, 7, 3, 1, 2, 6, 6, 6, 5, 5, 1, 6, 9, 3, 9,
           1, 9, 4, 1])

    >>> flatten([a, b, c, d], order='F', returns_shapes=False)
    ... # doctest: +NORMALIZE_WHITESPACE
    array([9, 4, 7, 4, 5, 2, 7, 2, 3, 6, 1, 6, 6, 3, 1, 9, 5, 9, 6, 4,
           5, 1, 9, 1])

    >>> w, x, y, z = unflatten(*flatten([a, b, c, d]))

    >>> w == a
    True

    >>> np.array_equal(x, b)
    True

    >>> np.array_equal(y, c)
    True

    >>> np.array_equal(z, d)
    True

    """
    ravel = partial(np.ravel, order=order)
    flattened = np.hstack(map(ravel, arys))

    if returns_shapes:
        shapes = list(map(np.shape, arys))
        return flattened, shapes

    return flattened


def unflatten(ary, shapes, order='C'):
    """
    Given a flat (1d) array, and a list of shapes (represented as tuples),
    return a list of ndarrays with the specified shapes.

    Parameters
    ----------
    ary : a 1d array
        A flat (1d) array.

    shapes : list of tuples
        A list of ndarray shapes (tuple of array dimensions)

    order : {'C', 'F', 'A'}, optional
        Reshape array using index order: C (row-major), Fortran
        (column-major) order, or preserve the C/Fortran ordering.
        The default is 'C'.

    Returns
    -------
    list of ndarrays
        A list of ndarrays with the specified shapes.

    See Also
    --------
    revrand.utils.flatten : its inverse

    Notes
    -----
    Equivalent to::

        lambda ary, shapes, order='C': \
            map(partial(custom_reshape, order=order),
                np.hsplit(ary, np.cumsum(map(partial(np.prod, dtype=int),
                                             shapes))), shapes)

    Examples
    --------

    >>> a = np.array([7, 4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3])

    >>> list(unflatten(a, [(1,), (1,), (4,), (2, 3)]))
    ... # doctest: +NORMALIZE_WHITESPACE
    [array([7]), array([4]), array([5, 8, 9, 1]), array([[4, 2, 5],
        [3, 4, 3]])]

    >>> list(unflatten(a, [(), (1,), (4,), (2, 3)]))
    ... # doctest: +NORMALIZE_WHITESPACE
    [7, array([4]), array([5, 8, 9, 1]), array([[4, 2, 5], [3, 4, 3]])]

    Fortran-order:

    >>> list(unflatten(a, [(1,), (1,), (4,), (2, 3)], order='F'))
    ... # doctest: +NORMALIZE_WHITESPACE
    [array([7]), array([4]), array([5, 8, 9, 1]), array([[4, 5, 4],
        [2, 3, 3]])]

    >>> list(unflatten(a, [(), (1,), (4,), (2, 3)], order='F'))
    ... # doctest: +NORMALIZE_WHITESPACE
    [7, array([4]), array([5, 8, 9, 1]), array([[4, 5, 4], [2, 3, 3]])]

    >>> list(unflatten(a, [(), (1,), (3,), (2, 3)]))
    ... # doctest: +NORMALIZE_WHITESPACE
    [7, array([4]), array([5, 8, 9]), array([[1, 4, 2], [5, 3, 4]])]

    >>> list(unflatten(a, [(), (1,), (5,), (2, 3)]))
    ... # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: total size of new array must be unchanged

    >>> flatten(list(unflatten(a, [(), (1,), (4,), (2, 3)])))
    ... # doctest: +NORMALIZE_WHITESPACE
    (array([7, 4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3]),
        [(), (1,), (4,), (2, 3)])
    """
    # important to make sure dtype is int
    # since prod on empty tuple is a float (1.0)
    sizes = list(map(partial(np.prod, dtype=int), shapes))
    sections = np.cumsum(sizes)
    subarrays = np.hsplit(ary, sections)
    # Subtle but important: last element of subarrays is always a extraneous
    # empty array but is ignored when zipped with shapes. Not really a bug...
    return map(partial(custom_reshape, order=order), subarrays, shapes)


def custom_reshape(a, newshape, order='C'):
    """
    Identical to `numpy.reshape` except in the case where `newshape` is
    the empty tuple, in which case we return a scalar instead of a
    0-dimensional array.

    Examples
    --------
    >>> a = np.arange(6)
    >>> np.array_equal(np.reshape(a, (3, 2)), custom_reshape(a, (3, 2)))
    True

    >>> custom_reshape(np.array([3.14]), newshape=())
    3.14

    >>> custom_reshape(np.array([2.71]), newshape=(1,))
    array([ 2.71])
    """
    if newshape == ():
        return np.asscalar(a)

    return np.reshape(a, newshape, order)


def map_indices(fn, iterable, indices):
    """
    Map a function across indices of an iterable

    Notes
    -----
    Roughly equivalent to, though more efficient than::

        lambda fn, iterable, *indices: (fn(arg) if i in indices else arg
                                        for i, arg in enumerate(iterable))

    Examples
    --------

    >>> a = [4, 6, 7, 1, 6, 8, 2]

    >>> from operator import mul
    >>> list(map_indices(partial(mul, 3), a, [0, 3, 5]))
    [12, 6, 7, 3, 6, 24, 2]

    >>> b = [9., np.array([5., 6., 2.]),
    ...      np.array([[5., 6., 2.], [2., 3., 9.]])]

    >>> list(map_indices(np.log, b, [0, 2])) # doctest: +NORMALIZE_WHITESPACE
    [2.1972245773362196,
     array([ 5.,  6.,  2.]),
     array([[ 1.60943791,  1.79175947,  0.69314718],
            [ 0.69314718,  1.09861229,  2.19722458]])]

    .. todo::

       Floating point precision

    >>> list(map_indices(np.exp, list(map_indices(np.log, b, [0, 2])), [0, 2]))
    ... # doctest: +NORMALIZE_WHITESPACE +SKIP
    [9.,
     array([5., 6., 2.]),
     array([[ 5.,  6.,  2.],
            [ 2.,  3.,  9.]])]
    """
    index_set = set(indices)
    for i, arg in enumerate(iterable):
        if i in index_set:
            yield fn(arg)
        else:
            yield arg
