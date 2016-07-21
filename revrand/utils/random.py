import numpy as np
import numbers

from sklearn.utils.validation import check_random_state


def endless_permutations(N, random_state=None):
    """
    Generate an endless sequence of random integers from permutations of the
    set [0, ..., N).

    If we call this N times, we will sweep through the entire set without
    replacement, on the (N+1)th call a new permutation will be created, etc.

    Parameters
    ----------
    N: int
        the length of the set
    random_state: int or RandomState, optional
        random seed

    Yields
    ------
    int:
        a random int from the set [0, ..., N)
    """
    generator = check_random_state(random_state)
    while True:
        batch_inds = generator.permutation(N)
        for b in batch_inds:
            yield b
