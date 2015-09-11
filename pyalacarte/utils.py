""" 
Reusable utility functions
"""

import numpy as np

from six.moves import map, range, reduce, zip
from itertools import chain, tee
from functools import partial
from operator import mul

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

def flatten(lst, order='C', returns_shapes=True):
    """
    Flatten ndarrays from a list of numpy scalars and/or ndarrays of possibly
    heterogenous dimensions and chain together into a flat (1D) list.

    .. note::

       Not to be confused with `np.ndarray.flatten()` (a more befitting might 
       be `chain` or maybe something else entirely since this function 
       is more than merely `chain` or `np.flatten`. Rather, it is the 
       composition of the former with the latter.

    Parameters
    ----------
    lst : list
        A list of scalars and/or numpy arrays of possibly heterogenous dimensions.

    order : {‘C’, ‘F’, ‘A’}, optional
        Whether to flatten in C (row-major), Fortran (column-major) order, 
        or preserve the C/Fortran ordering from a. The default is ‘C’.
    
    returns_shapes : bool, optional 
        Default is `True`. If `True`, the tuple (flattened, shapes) is returned,
        otherwise only the flattened is returned.

    Returns
    -------

    .. todo:: 

       For consistency, might consider keeping with the Python 3 theme of returning
       generators everywhere... Especially since most other functions here does...

    flattened,[shapes] : {list of numeric, list of tuples}
        Return the flat (1D) list chained together from flattened (according to order)
        ndarrays. When `returns_shapes` is `True`, return a list of tuples containing 
        also the shapes of each element of `lst` the second element.

    See Also
    --------
    pyalacarte.utils.unflatten : its inverse

    Notes
    -----
    Roughly equivalent to::

        def flatten(lst, order='C'):
            lsts, shapes = zip(*map(lambda x: (np.ravel(x, order), np.shape(x)), lst))
            return list(chain(*lsts)), shapes

    This implementation relies on the fact that scalars are 0-dimensional arrays. 
    That is,

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

       When 0-dimensional arrays of the latter form are flattened, *they  will be 
       unflattened as a scalar*. (Special cases aren't special enough to break the rules!)

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
    
    >>> flatten([a, b, c, d])
    ([9, 4, 7, 4, 5, 2, 7, 3, 1, 2, 6, 6, 6, 5, 5, 1, 6, 9, 3, 9, 1, 9, 4, 1], [(), (5,), (2, 3), (2, 2, 3)])

    >>> flatten([a, b, c, d], order='F')
    ([9, 4, 7, 4, 5, 2, 7, 2, 3, 6, 1, 6, 6, 3, 1, 9, 5, 9, 6, 4, 5, 1, 9, 1], [(), (5,), (2, 3), (2, 2, 3)])

    >>> flatten([a, b, c, d], returns_shapes=False)
    [9, 4, 7, 4, 5, 2, 7, 3, 1, 2, 6, 6, 6, 5, 5, 1, 6, 9, 3, 9, 1, 9, 4, 1]

    >>> flatten([a, b, c, d], order='F', returns_shapes=False)
    [9, 4, 7, 4, 5, 2, 7, 2, 3, 6, 1, 6, 6, 3, 1, 9, 5, 9, 6, 4, 5, 1, 9, 1]

    """
    ravel = partial(np.ravel, order=order)
    flattened = list(chain(*map(ravel, lst)))

    if returns_shapes:
        shapes = list(map(np.shape, lst))
        return flattened, shapes
    else:
        return flattened

def slices(sizes):
    """
    Notes
    -----
    Roughly equivalent to::

        lambda sizes: pairwise([0]+cumsum(sizes))

    Examples
    --------

    >>> list(slices([1, 3, 2, 3]))
    [(0, 1), (1, 4), (4, 6), (6, 9)]

    >>> list(slices([2, 0, 1]))
    [(0, 2), (2, 3)]

    >>> list(slices([2]))
    [(0, 2)]

    >>> list(slices([]))
    []
    """
    start = 0
    for size in sizes:
        if size == 0: 
            continue
        yield (start, start+size)
        start += size

def chunks(lst, sizes):
    """
    Notes
    -----

    Equivalent to::

        lambda lst, sizes: (lst[start:end] for start, end in slices(sizes))

    Examples
    --------

    >>> a = [2, 4, 6, 3, 4, 6, 1, 9, 3]

    >>> list(chunks(a, [3, 2, 4]))
    [[2, 4, 6], [3, 4], [6, 1, 9, 3]]

    >>> list(chunks([], [3, 2, 5]))
    [[], [], []]

    >>> list(chunks(a, []))
    []

    Lists are chunked greedily:

    >>> list(chunks(a, [3, 2, 3]))
    [[2, 4, 6], [3, 4], [6, 1, 9]]

    >>> list(chunks(a, [3, 2, 5]))
    [[2, 4, 6], [3, 4], [6, 1, 9, 3]]

    >>> list(chunks(a, [3, 1, 4]))
    [[2, 4, 6], [3], [4, 6, 1, 9]]

    >>> list(chunks(a, [3, 3, 4]))
    [[2, 4, 6], [3, 4, 6], [1, 9, 3]]

    """
    for start, end in slices(sizes):
        yield lst[start:end]

def unflatten(flat_lst, shapes, order='C'):
    """
    Given a flat (one-dimensional) list, and a list of ndarray shapes return 
    a list of numpy ndarrays of specified shapes.

    Parameters
    ----------
    flat_lst : list
        A flat (one-dimensional) list
    
    shapes : list of tuples
        A list of ndarray shapes (tuple of array dimensions)

    order : {‘C’, ‘F’, ‘A’}, optional
        Reshape array using index order: C (row-major), Fortran (column-major) 
        order, or preserve the C/Fortran ordering from a. The default is ‘C’.
    
    Returns
    -------
    list of ndarrays
        A list of numpy ndarrays of specified shapes 

    See Also
    --------
    pyalacarte.utils.flatten : its inverse

    Notes
    -----
    Equivalent to::

        lambda flat_lst, shapes, order='C': (np.reshape(chunk, shape, order) if shape else chunk[0] for 
            chunk, shape in zip(chunks(flat_lst, map(partial(np.prod, dtype=int), shapes)), shapes))

    Examples
    --------
    >>> list(unflatten([4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(2,), (3,), (2, 3)])) # doctest: +NORMALIZE_WHITESPACE
    [array([4, 5]), array([8, 9, 1]), array([[4, 2, 5], [3, 4, 3]])]

    >>> list(unflatten([7, 4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(), (1,), (4,), (2, 3)])) # doctest: +NORMALIZE_WHITESPACE
    [7, array([4]), array([5, 8, 9, 1]), array([[4, 2, 5], [3, 4, 3]])]

    >>> list(unflatten([4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(2,), (3,), (2, 3)], order='F')) # doctest: +NORMALIZE_WHITESPACE
    [array([4, 5]), array([8, 9, 1]), array([[4, 5, 4], [2, 3, 3]])]

    >>> list(unflatten([7, 4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(), (1,), (4,), (2, 3)], order='F')) # doctest: +NORMALIZE_WHITESPACE
    [7, array([4]), array([5, 8, 9, 1]), array([[4, 5, 4], [2, 3, 3]])]
    
    It goes without saying that `unflatten` is the inverse of of `flatten`. 
    For all lists `lst`, `unflatten(*flatten(lst)) == lst` and for all `shapes`
    `flatten(unflatten(lst, shapes)) == (lst, shapes)`

    >>> lst = [4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3]
    >>> shapes = [(2,), (3,), (2, 3)]

    >>> flatten(list(unflatten(lst, shapes))) == (lst, shapes)
    True

    .. todo::

       * Demonstrate that lists are unflattened greedily as well
       * Edge cases...

    """
    # important to make sure dtype is int
    # since prod on empty tuple is a float (1.0)
    sizes = map(partial(np.prod, dtype=int), shapes) 
    for chunk, shape in zip(chunks(flat_lst, sizes), shapes):
        if shape == ():
            # chunk only has 1 element
            yield from chunk
        else:
            yield np.reshape(chunk, shape, order)

def map_indices(fn, iterable, indices):
    
    """
    Notes
    -----
    Roughly equivalent to, though more efficient than::

        lambda fn, iterable, *indices: (fn(arg) if i in indices else arg for i, arg in enumerate(iterable))

    Examples
    --------

    >>> a = [4, 6, 7, 1, 6, 8, 2]

    >>> list(map_indices(partial(mul, 3), a, [0, 3, 5]))
    [12, 6, 7, 3, 6, 24, 2]

    >>> b = [9., np.array([5., 6., 2.]), np.array([[5., 6., 2.], [2., 3., 9.]])]
    
    >>> list(map_indices(np.log, b, [0, 2])) # doctest: +NORMALIZE_WHITESPACE
    [2.1972245773362196, 
     array([ 5.,  6.,  2.]), 
     array([[ 1.60943791,  1.79175947,  0.69314718],
            [ 0.69314718,  1.09861229,  2.19722458]])]

    .. todo::

       Floating point precision

    >>> list(map_indices(np.exp, list(map_indices(np.log, b, [0, 2])), [0, 2]))
    ... # doctest: +NORMALIZE_WHITESPACE
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

class CatParameters(object):

    def __init__(self, params, log_indices=None):

        self.pshapes = [np.asarray(p).shape if not np.isscalar(p)
                        else 1 for p in params]
        
        if log_indices is not None:
            self.log_indices = log_indices
        else:
            self.log_indices = []

    def flatten(self, params):
        """ This will take a list of parameters of scalars or arrays, and
            return a flattened array which is a concatenation of all of these
            parameters.

            This could be useful for using with an optimiser!

            Arguments:
                params: a list of scalars of arrays.

            Returns:
                list: a list or 1D array of scalars which is a flattened
                    concatenation of params.
        """

        vec = []
        for i, p in enumerate(params):
            fp = np.atleast_1d(p).flatten()
            vec.extend(fp if i not in self.log_indices else np.log(fp))

        return np.array(vec)

    def flatten_grads(self, params, grads):

        vec = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = np.atleast_1d(g)

            # Chain rule if log params used
            if i in self.log_indices:
                g *= np.atleast_1d(p)

            vec.extend(g.flatten())

        return np.array(vec)

    def unflatten(self, flatparams):
        """ This will turn a flattened list of parameters into the original
            parameter argument list, given a template.

            This could be useful for using after an optimiser!

            Argument:
                params: the template list of parameters.

                flatlist: the flattened list of parameters to turn into the
                    original parameter list.

            Returns:
                list: A list of the same form as params, but with the values
                    from flatlist.
        """

        rparams = []
        listind = 0
        for i, p in enumerate(self.pshapes):
            if np.isscalar(p):
                up = flatparams[listind]
                listind += 1
            else:
                nelems = np.product(p)
                up = np.reshape(flatparams[listind:(listind + nelems)], p)
                listind += nelems

            rparams.append(up if i not in self.log_indices else np.exp(up))

        return rparams


def params_to_list(params):
    """ This will take a list of parameters of scalars or arrays, and return a
        flattened array which is a concatenation of all of these parameters.

        This could be useful for using with an optimiser!

        Arguments:
            params: a list of scalars of arrays.

        Returns:
            list: a list or 1D array of scalars which is a flattened
                concatenation of params.
    """

    vec = []
    for p in params:
        vec.extend(np.atleast_1d(p).flatten())

    return vec


def list_to_params(params, flatlist):
    """ This will turn a flattened list of parameters into the original
        parameter argument list, given a template.

        This could be useful for using after an optimiser!

        Argument:
            params: the template list of parameters.

            flatlist: the flattened list of parameters to turn into the
                original parameter list.

        Returns:
            list: A list of the same form as params, but with the values from
                flatlist.
    """

    rparams = []
    listind = 0
    for p in params:
        if np.isscalar(p):
            rparams.append(flatlist[listind])
            listind += 1
        else:
            p = np.asarray(p)
            nelems = np.product(p.shape)
            rparams.append(np.reshape(flatlist[listind:(listind + nelems)],
                                      p.shape))
            listind += nelems

    return rparams
