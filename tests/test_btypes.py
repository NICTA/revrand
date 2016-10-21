"""Test revrand's bound and parameter types."""

from __future__ import division

import numpy as np

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

    v = 1.
    p = Parameter(v, Positive())
    assert p.value == v
    assert p.bounds.lower > 0
    assert p.bounds.upper is None
    # assert np.shape(p.rvs()) == np.shape(v)

    # v = np.ones(10)
    # p = Parameter(v, Bound())
    # assert np.shape(p.rvs()) == np.shape(v)

    # p = Parameter(v, Positive())
    # assert all(p.rvs() > 0)
