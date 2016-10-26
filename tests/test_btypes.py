"""Test revrand's bound and parameter types."""

from __future__ import division

import numpy as np
from scipy.stats import gamma

from revrand import Bound, Positive, Parameter


def test_bound():

    b = Bound(1, 2)
    assert b.lower == 1
    assert b.upper == 2

    assert b.check(1) is True
    assert b.check(3) is False
    assert b.clip(5) == 2

def test_positive():

    b = Positive(2)
    assert b.lower > 0
    assert b.upper == 2

    assert b.check(1) is True
    assert b.check(-1) is False
    assert b.clip(-3) == b.lower

def test_parameter():

    # Test "Null" parameter
    p = Parameter()
    assert p.shape == (0,)
    assert p.rvs() == []
    assert p.has_value is False
    assert p.is_random is False

    # Test values
    v = 1.
    p = Parameter(v, Positive())
    assert p.value == v
    assert p.bounds.lower > 0
    assert p.bounds.upper is None
    assert p.rvs() == v
    assert p.has_value is True
    assert p.is_random is False

    # Test distributions
    p = Parameter(gamma(1), Positive())
    assert np.shape(p.rvs()) == ()
    assert p.has_value is True
    assert p.is_random is True

    p = Parameter(gamma(1), Positive(), shape=(2,))
    assert np.shape(p.rvs()) == (2,)
    assert Positive().check(p.rvs())

    p = Parameter(gamma(1), Bound(1, 2), shape=(10, 5))
    assert np.shape(p.rvs()) == (10, 5)
    assert Bound(1, 2).check(p.rvs())
