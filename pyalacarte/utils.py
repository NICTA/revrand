""" 
Reusable utility functions
"""

import numpy as np

from six.moves import range, zip
from functools import partial
from itertools import tee

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
        
    First n iterators are created::

        iters = tee(iterable, n)

    Next, iterator i is advanced i times::

        for i, it in enumerate(iters):
            for _ in range(i):
                next(it, None)

    Then, the iterators are zipped back up again::

        return zip(*iters)

    Examples
    --------
    >>> a = [2, 5, 7, 4, 2, 8, 6]

    >>> list(nwise(a, n=3))
    [(2, 5, 7), (5, 7, 4), (7, 4, 2), (4, 2, 8), (2, 8, 6)]

    >>> pairwise = partial(nwise, n=2)
    >>> pairwise(a)
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

    >>> len(a) - 2 == len(list(nwise(a, 2))) - 1
    True

    >>> len(a) - 3 == len(list(nwise(a, 3))) - 1
    True

    >>> len(a) - 7 == len(list(nwise(a, 7))) - 1
    True
    """

    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        for _ in range(i):
            next(it, None)
    return zip(*iters)

pairwise = partial(nwise, n=2)

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
    Roughly equivalent to::

        lambda flat_lst, shapes: [np.asarray(flat_lst[start:end]).reshape(shape)
        for (start, end), shape in zip(pairwise(np.cumsum([0]+list(map(np.prod, shapes)))),
        shapes)]

    Examples
    --------
    >>> unflatten([4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(2,), (3,), (2, 3)])
    [array([4, 5]), array([8, 9, 1]), array([[4, 2, 5], [3, 4, 3]])]

    >>> unflatten([7, 4, 5, 8, 9, 1, 4, 2, 5, 3, 4, 3], [(,), (1,), (4,), (2, 3)])
    [7, array([4]), array([5, 8, 9, 1]), array([[4, 2, 5], [3, 4, 3]])]
    """
    for shape in shapes:
        pass

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
