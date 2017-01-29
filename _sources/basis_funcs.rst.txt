.. _basis_functions:

Basis Functions
===============

This module contains basis function objects that can be used with the machine
learning algorithms in *revrand*. It also implements basis *concatenation*
(kernel addition) using the ``Basis`` and ``BasisCat`` classes. See the
:ref:`quickstart` for an overview of the usage of these objects. 

.. note::
    When calling ``transform`` or ``grad`` on concatenated bases while using
    arguments, be careful of ordering. For instance,

        >>> from revrand.basis_functions import RandomRBF, RandomMatern52
        >>> base = RandomRBF(Xdim=D, nbases=n) \
        ...     + RandomMatern52(Xdim=D, nbases=n)
        >>> base.transform(X)

    This call to ``transform`` just uses the default value for the length 
    scales in both bases.

        >>> base.transform(X, 1)

    Will pass a length scale of 1 to just the *first* basis object
    (``RandomRBF``), and the second (``RandomMatern52``) will use its default.

        >>> base.transform(X, None, 2)

    Will make the first basis object use its default, and make the *second* use
    a value of 2.

        >>> base.transform(X, 1, 2)

    Will make the first use a value of 1, and the second, 2.

.. currentmodule:: revrand.basis_functions

.. autosummary::
    :toctree: generated/

    Basis
    BiasBasis
    LinearBasis
    PolynomialBasis
    RadialBasis
    SigmoidalBasis
    RandomRBF
    RandomLaplace
    RandomCauchy
    RandomMatern32
    RandomMatern52
    OrthogonalRBF
    FastFoodRBF
    FastFoodGM

.. automodule:: revrand.basis_functions
    :members:
    :noindex:
